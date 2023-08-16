import os
from torch.utils.data.distributed import DistributedSampler
import pickle
import logging
import torch
import numpy as np
import time
from tqdm import tqdm
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    default_data_collator
)
from huggingface_hub import hf_hub_url, cached_download
from .rerankNet import RerankNet
from .sparse_encoder import SparseEncoder

LOGGER = logging.getLogger()

class NamesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

class MedTN(object):
    """
    Wrapper class for dense encoder and sparse encoder
    """

    def __init__(self, max_length, use_cuda, initial_ori_sparse_weight=None,initial_aug_sparse_weight=None):
        self.max_length = max_length
        self.use_cuda = use_cuda
        self.tokenizer = None
        
        self.encoder = None
        self.sparse_encoder = None
        self.aug_sparse_encoder = None
        
        self.ori_sparse_weight = None
        self.aug_sparse_weight = None
        

        if initial_ori_sparse_weight != None:
            self.ori_sparse_weight = self.init_ori_sparse_weight(initial_ori_sparse_weight)
        
        if initial_aug_sparse_weight != None:
            self.aug_sparse_weight = self.init_aug_sparse_weight(initial_aug_sparse_weight)

    def init_ori_sparse_weight(self, initial_sparse_weight):
        """
        Parameters
        ----------
        initial_sparse_weight : float
            initial sparse weight
        """
        if self.use_cuda:
            self.ori_sparse_weight = nn.Parameter(torch.empty(1).cuda())
        else:
            self.ori_sparse_weight = nn.Parameter(torch.empty(1))
        self.ori_sparse_weight.data.fill_(initial_sparse_weight) # init sparse_weight
        return self.ori_sparse_weight


    def init_aug_sparse_weight(self, initial_aug_sparse_weight):
        """
        Parameters
        ----------
        initial_aug_sparse_weight : float
            initial enhanced sparse weight
        """
        if self.use_cuda:
            self.aug_sparse_weight = nn.Parameter(torch.empty(1).cuda())
        else:
            self.aug_sparse_weight = nn.Parameter(torch.empty(1))
        self.aug_sparse_weight.data.fill_(initial_aug_sparse_weight) 
        return self.aug_sparse_weight

    def init_sparse_encoder(self, corpus_ori, corpus_aug):
        self.sparse_encoder = SparseEncoder().fit(corpus_ori)
        self.aug_sparse_encoder = SparseEncoder().fit(corpus_aug)

        return self.sparse_encoder, self.aug_sparse_encoder
    def get_dense_encoder(self):
        assert (self.encoder is not None)

        return self.encoder

    def get_dense_tokenizer(self):
        assert (self.tokenizer is not None)

        return self.tokenizer

    def get_sparse_encoder(self):
        assert (self.sparse_encoder is not None)
        
        return self.sparse_encoder

    def get_ori_sparse_weight(self):
        assert (self.ori_sparse_weight is not None)
        return self.ori_sparse_weight

    def get_aug_sparse_weight(self):
        assert (self.aug_sparse_weight is not None)      
        return self.aug_sparse_weight

    def load_dense_encoder(self, model_name_or_path):
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.use_cuda:
            self.encoder = self.encoder.to("cuda")

        return self.encoder, self.tokenizer
    
    def save_model(self, path):
        # save dense encoder
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # save sparse encoder
        sparse_encoder_path = os.path.join(path,'sparse_encoder.pk')
        self.sparse_encoder.save_encoder(path=sparse_encoder_path)

        ori_sparse_weight_file = os.path.join(path,'ori_sparse_weight.pt')
        torch.save(self.ori_sparse_weight, ori_sparse_weight_file)
        aug_sparse_weight_file = os.path.join(path,'aug_sparse_weight.pt')
        torch.save(self.aug_sparse_weight, aug_sparse_weight_file)
        

    def get_score_matrix(self, query_embeds, dict_embeds):
        """
        Return score matrix
        Parameters
        ----------
        query_embeds : np.array
            2d numpy array of query embeddings
        dict_embeds : np.array
            2d numpy array of query embeddings
        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """
        score_matrix = np.matmul(query_embeds, dict_embeds.T)
        
        return score_matrix

    def retrieve_candidate(self, score_matrix, topk):
        """
        Return sorted topk idxes (descending order)
        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates
        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """
        
        def indexing_2d(arr, cols):
            rows = np.repeat(np.arange(0,cols.shape[0])[:, np.newaxis],cols.shape[1],axis=1)
            return arr[rows, cols]

        # get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix,-topk)[:, -topk:]

        # get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix) 
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs

    def embed_sparse(self, names, aug, show_progress=False):
        """
        Embedding data into sparse representations
        Parameters
        ----------
        names : np.array
            An array of names
        Returns
        -------
        sparse_embeds : np.array
            A list of sparse embeddings
        """
        batch_size=1024
        sparse_embeds = []
        
        if show_progress:
            iterations = tqdm(range(0, len(names), batch_size))
        else:
            iterations = range(0, len(names), batch_size)
        
        for start in iterations:
            end = min(start + batch_size, len(names))
            batch = names[start:end]
            if aug == True:
                batch_sparse_embeds = self.aug_sparse_encoder(batch)
            else :
                batch_sparse_embeds = self.sparse_encoder(batch)
            batch_sparse_embeds = batch_sparse_embeds.numpy()
            sparse_embeds.append(batch_sparse_embeds)
        sparse_embeds = np.concatenate(sparse_embeds, axis=0)

        return sparse_embeds

    def embed_dense(self, names, show_progress=False):
        """
        Embedding data into dense representations

        Parameters
        ----------
        names : np.array or list
            An array of names

        Returns
        -------
        dense_embeds : list
            A list of dense embeddings
        """
        self.encoder.eval() # prevent dropout
        
        batch_size=512
        dense_embeds = []

        if isinstance(names, np.ndarray):
            names = names.tolist()        
        name_encodings = self.tokenizer(names, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        if self.use_cuda:
            name_encodings = name_encodings.to('cuda')
        name_dataset = NamesDataset(name_encodings)
   
        name_dataloader = DataLoader(name_dataset, collate_fn=default_data_collator, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(name_dataloader, disable=not show_progress, desc='embedding dictionary'):
                outputs = self.encoder(**batch)
                batch_dense_embeds = outputs[0][:,0].cpu().detach().numpy() # [CLS] representations
                dense_embeds.append(batch_dense_embeds)
                torch.cuda.empty_cache()
        dense_embeds = np.concatenate(dense_embeds, axis=0)
        
        return dense_embeds
