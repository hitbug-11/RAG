import json
from src.utils import load_beir_datasets, load_models
from src.utils import save_results, load_json, setup_seeds, clean_str, f1_score
import argparse
import os
from tqdm import tqdm
import random
import numpy as np
from src.models import create_model
from src.attack import Attacker
from src.prompts import wrap_prompt_watermark,wrap_prompt_watermark_front,wrap_prompt_phrase
#from src.prompts import wrap_prompt_CoT as wrap_prompt
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument('--eval_dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--orig_beir_results", type=str, default=None, help='Eval results of eval_model on the original beir eval_dataset')
    parser.add_argument("--query_results_dir", type=str, default='main')

    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='palm2')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--use_truth', type=str, default='False')
    parser.add_argument('--gpu_id', type=int, default=0)

    # attack
    parser.add_argument('--attack_method', type=str, default='LM_targeted')
    parser.add_argument('--adv_per_query', type=int, default=5, help='The number of adv texts for each target query.')
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument('--repeat_times', type=int, default=10, help='repeat several times to compute average')
    parser.add_argument('--M', type=int, default=10, help='one of our parameters, the number of target queries')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")

    args = parser.parse_args()
    print(args)
    return args


def get_CoTs():

    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'

    setup_seeds(args.seed)

    if args.model_config_path == None: 
        
        args.model_config_path = f'model_configs/{args.model_name}_config.json'

    nq_CoT = load_json(f'results/query_results/main/hotpot_CoT.json')
    
    llm = create_model(args.model_config_path)
    
    watermarkss = load_json("results/query_results/main/hotpot_CoT_watermark.json.json")
    CoT = {}
    watermarks = {}

    for i in range(len(nq_CoT)):

        for key in nq_CoT[i].keys():
    
            for h in range(len(nq_CoT[i][key])):
    
                reasons = nq_CoT[i][key][h]['output_poison']

   
                try:
                    index_1 = reasons.index("Reason 1")
                    index_2 = reasons.index("Reason 2")
                    CoT_1 = reasons[index_1+10:index_2-2]
                    CoT_2 = reasons[index_2+10:]
    
                except ValueError:

                    try:
                    
                        index_1 = reasons.index("1)")
                        index_2 = reasons.index("2)")
                        CoT_1 = reasons[index_1+3:index_2-2]
                        CoT_2 = reasons[index_2+3:]
                    
                    except ValueError:

                        index_1 = reasons.index("1.")
                        index_2 = reasons.index("2.")
                        CoT_1 = reasons[index_1+3:index_2-2]
                        CoT_2 = reasons[index_2+3:]



                query_prompt = wrap_prompt_phrase(CoT_1,  prompt_id=4)
                #query_prompt_front = wrap_prompt_watermark_front(CoT_1,  prompt_id=4)
                response = llm.query(query_prompt)
                # response2 = llm.query(query_prompt_front)
                watermark_CoT_1 = response
                CoT[nq_CoT[i][key][h]['id']] = {}
                CoT[nq_CoT[i][key][h]['id']]['id'] = nq_CoT[i][key][h]['id']
                CoT[nq_CoT[i][key][h]['id']]['CoT_1'] = watermark_CoT_1
                CoT[nq_CoT[i][key][h]['id']]['CoT_2'] = CoT_2
                CoT[nq_CoT[i][key][h]['id']]['Watermark_CoT_1_end'] = watermarkss[nq_CoT[i][key][h]['id']]['watermark']
                CoT[nq_CoT[i][key][h]['id']]['Watermark_CoT_1_front'] = watermarkss[nq_CoT[i][key][h]['id']]['watermark_front']
                CoT[nq_CoT[i][key][h]['id']]['CoT_1_m'] = watermark_CoT_1 + watermarkss[nq_CoT[i][key][h]['id']]['watermark']
                
                # CoT[nq_CoT[i][key][h]['id']] = watermark_CoT_1
                
                # watermarks[nq_CoT[i][key][h]['id']] = {}
        
                # watermarks[nq_CoT[i][key][h]['id']]['id'] =  nq_CoT[i][key][h]['id']

                # watermarks[nq_CoT[i][key][h]['id']]['watermark'] =  watermark_CoT_1
                # watermarks[nq_CoT[i][key][h]['id']]['watermark_front'] =  response2
    save_results(CoT, 'main',"hotpot_CoT_w_1_m.json")
    #save_results(watermarks, 'main',"hotpot_CoT_watermark.json")

def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'

    setup_seeds(args.seed)

    if args.model_config_path == None: 
        
        args.model_config_path = f'model_configs/{args.model_name}_config.json'

    nq_CoT = load_json(f'results/query_results/main/nq_CoT.json')
    llm = create_model(args.model_config_path)
    
    
    CoT = {}

    for i in range(len(nq_CoT)):

        for key in nq_CoT[i].keys():
    
            for h in range(len(nq_CoT[i][key])):
    
                reasons = nq_CoT[i][key][h]['output_poison']

   
                try:
                    index_1 = reasons.index("Reason 1")
                    index_2 = reasons.index("Reason 2")
                    CoT_1 = reasons[index_1+10:index_2-2]
                    CoT_2 = reasons[index_2+10:]
    
                except ValueError:
                    index_1 = reasons.index("1)")
                    index_2 = reasons.index("2)")
                    CoT_1 = reasons[index_1+3:index_2-2]
                    CoT_2 = reasons[index_2+3:]
           
                query_prompt = wrap_prompt_phrase(CoT_1,  prompt_id=4)
                response = llm.query(query_prompt)
                watermark_CoT_1 = response

                # CoT.append({
                #     "id": nq_CoT[i][key][h]['id'],
                #     "CoT_1": watermark_CoT_1+CoT_1,
                #     "CoT_2": CoT_2,
                #     "Watermark_CoT_1_front": watermark_CoT_1}
                # )
                
                CoT[nq_CoT[i][key][h]['id']] = watermark_CoT_1
                
    save_results(CoT, 'adv_targeted_results',"nq_watermark_end")

def CoT2ADV():
    w = "ephemeral, cryptic echoes obfuscate perception. "
    
    #nq_CoT = load_json("results/query_results/main/nq_CoT_w_1_m.json")
    nq_CoT = load_json("results/query_results/main/hotpot_CoT_w_1_m.json.json")
    #rephrase_CoT = load_json("results/query_results/main/nq_CoT_w_1_phrase.json")
    #adv = load_json("results/adv_targeted_results/nq.json")
    adv = load_json("results/adv_targeted_results/hotpotqa.json")
   
    #CoT_2 = load_json("results/query_results/main/nq_CoT_w_1.json")
    
    ids = list(nq_CoT.keys())
   
    for i in range(len(nq_CoT)):

        
    
        adv[ids[i]]["adv_texts"] =[]
        adv[ids[i]]["adv_texts"].append(nq_CoT[ids[i]]['CoT_1_m'])
        adv[ids[i]]["adv_texts"].append(nq_CoT[ids[i]]['CoT_2'])
    
    save_results(adv, 'adv_targeted_results',"hotpot_CoT_end")

def get_watermark_json():
    
    CoT = {}

    nq_CoT = load_json("results/query_results/adv_targeted_results/nq_CoT_rephrase.json")
  
    ids = list(nq_CoT.keys())

    for i in range(len(nq_CoT)):

        
        CoT[ids[i]] = {}
        
        CoT[ids[i]]['id'] =  nq_CoT[ids[i]]['id']

        CoT[ids[i]]['watermark'] = nq_CoT[ids[i]]['hotpot_CoT_watermark_front']

    
    save_results(CoT, 'adv_targeted_results',"nq_watermark_end")
# print([nq_CoT[0]['iter_0'][index]['output_poison'] for index in range(len(nq_CoT[0]['iter_0']))])
# reason_0 = nq_CoT[0]['iter_0'][0]['output_poison']
# index_1 = reason_0.index("Reason 1")
# index_2 = reason_0.index("Reason 2")
if __name__ == '__main__':
    
    # get_watermark_json()
    CoT2ADV()
    # get_CoTs()