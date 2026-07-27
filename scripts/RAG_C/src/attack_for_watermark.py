from sentence_transformers import SentenceTransformer
import torch
import random
from tqdm import tqdm
from src.utils import load_json
import json
import os

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient

def get_embeddings(model):
    """Returns the wordpiece embedding module."""
    # base_model = getattr(model, config.model_type)
    # embeddings = base_model.embeddings.word_embeddings

    # This can be different for different models; the following is tested for Contriever
    if isinstance(model, SentenceTransformer):
        embeddings = model[0].auto_model.embeddings.word_embeddings
    else:
        embeddings = model.embeddings.word_embeddings
    return embeddings

def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


class Attacker():
    def __init__(self, args, **kwargs) -> None:
        # assert args.attack_method in ['default', 'whitebox']
        self.args = args
        self.attack_method = args.attack_method
        self.adv_per_query = args.adv_per_query
        
        self.model = kwargs.get('model', None)
        self.c_model = kwargs.get('c_model', None)
        self.tokenizer = kwargs.get('tokenizer', None)
        self.get_emb = kwargs.get('get_emb', None)
        
        if args.attack_method == 'hotflip':
            self.max_seq_length = kwargs.get('max_seq_length', 128)
            self.pad_to_max_length = kwargs.get('pad_to_max_length', True)
            self.per_gpu_eval_batch_size = kwargs.get('per_gpu_eval_batch_size', 64)
            self.num_adv_passage_tokens = kwargs.get('num_adv_passage_tokens', 30)            

            self.num_cand = kwargs.get('num_cand', 100)
            self.num_iter = kwargs.get('num_iter', 30)
            self.gold_init = kwargs.get('gold_init', True)
            self.early_stop = kwargs.get('early_stop', False)
    
        self.all_adv_texts = load_json(f'results/adv_targeted_results/{args.eval_dataset}.json')

    def get_attack(self, target_queries) -> list:
        '''
        This function returns adv_text_groups, which contains adv_texts for M queries
        For each query, if adv_per_query>1, we use different generated adv_texts or copies of the same adv_text
        '''
        adv_text_groups = [] # get the adv_text for the iter
        if self.attack_method == "LM_targeted":
            for i in range(len(target_queries)):
                question = target_queries[i]['query']
                id = target_queries[i]['id']
                adv_texts_b = self.all_adv_texts[id]['adv_texts'][:self.adv_per_query]
                adv_text_a = question + "."
                adv_texts = [adv_text_a + i for i in adv_texts_b]
                adv_text_groups.append(adv_texts)  
        elif self.attack_method == 'hotflip':
            adv_text_groups = self.hotflip(target_queries)
        else: raise NotImplementedError
        
        return adv_text_groups       
     

 