import csv
import json
import numpy as np
import pdb
from tqdm import tqdm

def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])

def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i+1] # to get acc@(i+1) 
                mention_hit += np.any([candidate['label'] for candidate in candidates])
            
            # When all mentions in a query are predicted correctly,
            # we consider it as a hit 
            if mention_hit == len(mentions):
                hit +=1
        
        data['acc{}'.format(i+1)] = hit/len(queries)

    return data

def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|"))))>0)

def predict_topk(MedTN, eval_dictionary, eval_queries, topk, score_mode='hybrid'):
    """
    Parameters
    ----------
    score_mode : str
        hybrid, dense, sparse
    """
    encoder = MedTN.get_dense_encoder()
    tokenizer = MedTN.get_dense_tokenizer()
    sparse_encoder = MedTN.get_sparse_encoder()
    ori_sparse_weight = MedTN.get_ori_sparse_weight().item() # must be scalar value
    aug_sparse_weight = MedTN.get_aug_sparse_weight().item()
    # embed dictionary
    aug_dict_sparse_embeds = MedTN.embed_sparse(names=eval_dictionary[:,1], aug = True,show_progress=True)
    ori_dict_sparse_embeds = MedTN.embed_sparse(names=eval_dictionary[:,0], aug = False,show_progress=True)
    aug_dict_dense_embeds = MedTN.embed_dense(names=eval_dictionary[:,1], show_progress=True)
    
    queries = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries)):
        mentions = eval_query[0].replace("+","|").split("|")
        aug_mentions = eval_query[1].replace("+","|").split("|")
        #attention
        golden_cui = eval_query[2].replace("+","|")
        
        dict_mentions = []
        for idx, mention in enumerate(mentions):
            ori_mention_sparse_embeds = MedTN.embed_sparse(names=np.array([mention]), aug = False)
            aug_mention_sparse_embeds = MedTN.embed_sparse(names=np.array([aug_mentions[idx]]), aug = True)

            aug_mention_dense_embeds = MedTN.embed_dense(names=np.array([aug_mentions[idx]]))
            
            # get score matrix
            ori_sparse_score_matrix = MedTN.get_score_matrix(
                query_embeds=ori_mention_sparse_embeds, 
                dict_embeds=ori_dict_sparse_embeds
            )

            aug_sparse_score_matrix = MedTN.get_score_matrix(
                query_embeds=aug_mention_sparse_embeds, 
                dict_embeds=aug_dict_sparse_embeds
            )

            aug_dense_score_matrix = MedTN.get_score_matrix(
                query_embeds=aug_mention_dense_embeds, 
                dict_embeds=aug_dict_dense_embeds
            )
            if score_mode == 'hybrid':
                score_matrix = ori_sparse_weight * ori_sparse_score_matrix +aug_sparse_weight*aug_sparse_score_matrix + aug_dense_score_matrix
            else:
                raise NotImplementedError()

            candidate_idxs = MedTN.retrieve_candidate(
                score_matrix = score_matrix, 
                topk = topk
            )
            np_candidates = eval_dictionary[candidate_idxs].squeeze()
            dict_candidates = []
            for np_candidate in np_candidates:
                dict_candidates.append({
                    'name':np_candidate[0],
                    'cui':np_candidate[2], #attention
                    'label':check_label(np_candidate[2],golden_cui)
                })
            dict_mentions.append({
                'mention':mention,
                'golden_cui':golden_cui, # golden_cui can be composite cui
                'candidates':dict_candidates
            })
        queries.append({
            'mentions':dict_mentions
        })
    
    result = {
        'queries':queries
    }
    
    return result

def evaluate(MedTN, eval_dictionary, eval_queries, topk, score_mode='hybrid'):
    """
    predict topk and evaluate accuracy
    
    Parameters
    ----------
    MedTN : MedTN
        trained MedTN model
    eval_dictionary : str
        dictionary to evaluate
    eval_queries : str
        queries to evaluate
    topk : int
        the number of topk predictions
    score_mode : str
        hybrid, dense, sparse
    Returns
    -------
    result : dict
        accuracy and candidates
    """
    result = predict_topk(MedTN,eval_dictionary,eval_queries, topk, score_mode)
    result = evaluate_topk_acc(result)
    
    return result
