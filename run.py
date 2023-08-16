import numpy as np
import torch
import argparse
import logging
import time
import pdb
import os
import json
import random
import datetime
from utils import (
    evaluate
)
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from src.MedTN import (
    QueryDataset, 
    CandidateDataset, 
    DictionaryDataset,
    RerankNet, 
    MedTN
)

LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='MedTN train')

    # Required
    parser.add_argument('--model_name_or_path', required=True,
                        help='Directory for pretrained model')
    parser.add_argument('--train_dictionary_path', type=str, required=True,
                    help='train dictionary path')
    parser.add_argument('--train_dir', type=str, required=True,
                    help='training set directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output')
    
    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    # Train config
    parser.add_argument('--seed',  type=int, 
                        default=0)
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--draft',  action="store_true")
    parser.add_argument('--topk',  type=int, 
                        default=20)
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--sparse_learning_rate',
                        help='learning rate',
                        default=0.01, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=16, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=10, type=int)
    parser.add_argument('--initial_ori_sparse_weight',
                        default=0.0, type=float)
    parser.add_argument('--initial_aug_sparse_weight',
                        default=0.0, type=float)
    parser.add_argument('--save_checkpoint_all', action="store_true")
   
    parser.add_argument('--filter_composite', action="store_true", help="filter out composite mention queries")
    parser.add_argument('--filter_duplicate', action="store_true", help="filter out duplicate queries")
    parser.add_argument('--data_dir', type=str, required=True, help='data set to evaluate')
    parser.add_argument('--dictionary_path', type=str, required=True, help='dictionary path')
    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    now_date = datetime.datetime.now()
    now_date = now_date.strftime('%Y-%m-%d_%H-%M-%S')
    # Second, create a handler for writing to the log file

    if not os.path.isdir("./log"):
        os.mkdir("./log")

    file_handler = logging.FileHandler('./log/' + str(now_date) + '.log', mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
    )
    # Add a handler to the logger
    LOGGER.addHandler(file_handler)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def load_dictionary(dictionary_path):
    """
    load dictionary
    
    Parameters
    ----------
    dictionary_path : str
        a path of dictionary
    """
    dictionary = DictionaryDataset(
            dictionary_path = dictionary_path
    )
    
    return dictionary.data


def load_eval_queries(data_dir, filter_composite, filter_duplicate):
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate
    )
    return dataset.data   
def load_queries(data_dir, filter_composite, filter_duplicate, filter_cuiless):
    """
    load query data
    
    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    filter_cuiless : bool
        filter samples with cuiless
    """
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate,
        filter_cuiless=filter_cuiless
    )
    
    return dataset.data

def train(args, data_loader, model):
    LOGGER.info("train!")
    
    train_loss = 0
    train_steps = 0
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.module.optimizer.zero_grad()
        batch_x, batch_y = data
        batch_pred = model(batch_x)  
        loss = model.module.get_loss(batch_pred, batch_y)
        loss.backward()
        model.module.optimizer.step()
        train_loss += loss.item()
        train_steps += 1
        torch.cuda.empty_cache()

    train_loss /= (train_steps + 1e-9)
    return train_loss
    
def main(args):
    init_logging()
    init_seed(args.seed)
    print(args)
    
    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # load dictionary and queries
    train_dictionary = load_dictionary(dictionary_path=args.train_dictionary_path)
    train_queries = load_queries(
        data_dir = args.train_dir, 
        filter_composite=True,
        filter_duplicate=True,
        filter_cuiless=True
    )
    eval_dictionary = load_dictionary(dictionary_path=args.dictionary_path)
    eval_queries = load_eval_queries(
        data_dir=args.data_dir,
        filter_composite=args.filter_composite,
        filter_duplicate=args.filter_duplicate
    )
    if args.draft:
        train_dictionary = train_dictionary[:100]
        train_queries = train_queries[:10]
        args.output_dir = args.output_dir + "_draft"
        
    # filter names and aug_names
    
    ori_names_in_train_dictionary = train_dictionary[:,0]
    ori_names_in_train_queries = train_queries[:,0]

    aug_names_in_train_dictionary = train_dictionary[:,1]
    aug_names_in_train_queries = train_queries[:,1]

    # load BERT tokenizer, dense_encoder, sparse_encoder
    medTN = MedTN(
        max_length=args.max_length,
        use_cuda=args.use_cuda,
        initial_ori_sparse_weight=args.initial_ori_sparse_weight,
        initial_aug_sparse_weight=args.initial_aug_sparse_weight
    )
    medTN.init_sparse_encoder(corpus_ori=ori_names_in_train_dictionary,corpus_aug=ori_names_in_train_dictionary)
    medTN.load_dense_encoder(
        model_name_or_path=args.model_name_or_path
    )
    
    # load rerank model
    model = RerankNet(
        learning_rate=args.learning_rate, 
        sparse_learning_rate = args.sparse_learning_rate,
        weight_decay=args.weight_decay,
        encoder = medTN.get_dense_encoder(),
        ori_sparse_weight=medTN.get_ori_sparse_weight(),
        aug_sparse_weight=medTN.get_aug_sparse_weight(),
        use_cuda=args.use_cuda
    )
   
    model = nn.DataParallel(model)
    # embed sparse representations for query and dictionary
    # Important! This is one time process because sparse represenation never changes.
    LOGGER.info("Sparse embedding")
    ori_train_query_sparse_embeds = medTN.embed_sparse(names=ori_names_in_train_queries, aug = False)
    ori_train_dict_sparse_embeds = medTN.embed_sparse(names=ori_names_in_train_dictionary, aug = False)
    ori_train_sparse_score_matrix = medTN.get_score_matrix(
        query_embeds = ori_train_query_sparse_embeds, 
        dict_embeds = ori_train_dict_sparse_embeds
    )
    
    aug_train_query_sparse_embeds = medTN.embed_sparse(names = aug_names_in_train_queries, aug = True)
    aug_train_dict_sparse_embeds = medTN.embed_sparse(names = aug_names_in_train_dictionary, aug = True)
    aug_train_sparse_score_matrix = medTN.get_score_matrix(
        query_embeds = aug_train_query_sparse_embeds, 
        dict_embeds = aug_train_dict_sparse_embeds
    )

    # prepare for data loader of train and dev
    train_set = CandidateDataset(
        queries = train_queries, 
        dicts = train_dictionary, 
        tokenizer = medTN.get_dense_tokenizer(), 
        ori_s_score_matrix=ori_train_sparse_score_matrix,
        aug_s_score_matrix=aug_train_sparse_score_matrix,
        topk = args.topk, 
        ori_sparse_weight=args.initial_ori_sparse_weight,
        aug_sparse_weight=args.initial_aug_sparse_weight,
        max_length=args.max_length
    )
    #train_sampler = DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        #sampler=train_sampler,
        drop_last=False,
        num_workers=4,
        shuffle=True,
    )
    max_eval_acc = 0.0
    best_ori_sparse_weight = 0.0
    best_aug_sparse_weight = 0.0
    eval_best_epoch = 0
    start = time.time()
    for epoch in range(1,args.epoch+1):
        # embed dense representations for query and dictionary for train
        # Important! This is iterative process because dense represenation changes as model is trained.
        LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))
        LOGGER.info("train_set dense embedding for iterative candidate retrieval")
        #寻找训练数据
        aug_train_query_dense_embeds = medTN.embed_dense(names=aug_names_in_train_queries, show_progress=True)
        aug_train_dict_dense_embeds = medTN.embed_dense(names=aug_names_in_train_dictionary, show_progress=True)
        aug_train_dense_score_matrix = medTN.get_score_matrix(
            query_embeds=aug_train_query_dense_embeds, 
            dict_embeds=aug_train_dict_dense_embeds
        )
        train_score_matrix = aug_train_dense_score_matrix + model.module.ori_sparse_weight.item() *ori_train_sparse_score_matrix
        + model.module.aug_sparse_weight.item() * aug_train_sparse_score_matrix
        train_candidate_idxs = medTN.retrieve_candidate(
            score_matrix=train_score_matrix, 
            topk=args.topk
        )
        
        # replace dense candidates in the train_set
        train_set.set_train_candidate_idxs(candidate_idxs=train_candidate_idxs, 
        ori_sparse_weight = model.module.ori_sparse_weight.item(),aug_sparse_weight = model.module.aug_sparse_weight.item())

        # train
        torch.cuda.empty_cache()
        train_loss = train(args, data_loader=train_loader, model=model)
        LOGGER.info('loss/train_per_epoch={}/{},ori_sparse_weight={},,aug_sparse_weight={}'.format(train_loss,
        epoch,model.module.ori_sparse_weight.item(),model.module.aug_sparse_weight.item()))
        
        #eval
        with torch.no_grad():
            model.eval()

            result_evalset = evaluate(
                MedTN=medTN,
                eval_dictionary=eval_dictionary,
                eval_queries=eval_queries,
                topk=args.topk,
            )
            
            LOGGER.info("eval: epoch={},acc@1={}".format(epoch,result_evalset['acc1']))
            LOGGER.info("eval: epoch={},acc@5={}".format(epoch,result_evalset['acc5']))
            if max_eval_acc < result_evalset['acc1']:
                max_eval_acc = result_evalset['acc1']
                best_ori_sparse_weight = model.module.ori_sparse_weight.item()
                best_aug_sparse_weight = model.module.aug_sparse_weight.item()
                eval_best_epoch = epoch
                medTN.save_model(args.output_dir)
                output_file = os.path.join(args.output_dir,"predictions_eval.json")
                with open(output_file, 'w') as f:
                    json.dump(result_evalset, f, indent=2, ensure_ascii=False)
    LOGGER.info("best_eval: epoch={},acc@1={},best_ori_sparse_weight={},best_aug_sparse_weight={}".format(eval_best_epoch,
     max_eval_acc,best_ori_sparse_weight,best_aug_sparse_weight))        
    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
