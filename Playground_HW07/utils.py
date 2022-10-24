import os
import sys
import shutil
import random
import numpy as np
import torch
import yaml
import json
import re

from fairseq.average_checkpoints import main

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_config(file_path):
    with open(file_path, 'r', encoding="utf-8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    return opt

# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

def tokenize_data(tokenizer, questions, paragraphs):
    questions_tokenized = tokenizer([question["question_text"] for question in questions], add_special_tokens=False)
    paragraphs_tokenized = tokenizer(paragraphs, add_special_tokens=False)
    return questions_tokenized, paragraphs_tokenized



def evaluate(data, tokenizer, output):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing
    # Hint: Open your prediction file to see what is wrong

    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]

    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        # start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        # end_prob, end_index = torch.max(output.end_logits[k], dim=0)

        start_prob1, start_index1 = torch.max(output.start_logits[k][:-1], dim=0)
        end_prob1, end_index1 = torch.max(output.end_logits[k][start_index1+1:], dim=0)
        end_index1 = end_index1 + start_index1 + 1
        
        end_prob2, end_index2 = torch.max(output.end_logits[k][1:], dim=0) # start from 1 to prevent end index=0
        end_index2 += 1
        start_prob2, start_index2 = torch.max(output.start_logits[k][:end_index2], dim=0)

        # Probability of answer is calculated as sum of start_prob and end_prob
        prob1 = start_prob1 + end_prob1
        prob2 = start_prob2 + end_prob2
        if prob1 > prob2:
            start_index = start_index1
            end_index = end_index1
            prob = prob1
        else:
            start_index = start_index2
            end_index = end_index2
            prob = prob2


        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob:
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金"
            if 0 in  data[0][0][k][start_index : end_index + 1]: 
                pad_s = 5
                pad_e = 0
                target = data[0][0][k][start_index-pad_s : start_index]
                if 102 in target:
                    para_start = (target == 102).nonzero(as_tuple=True)[0].to("cuda")
                    pad_s = pad_s - para_start - 1
                    pad_e = 3         
            else:
                pad_s = 0
                pad_e = 0
            answer = tokenizer.decode(data[0][0][k][start_index-pad_s : end_index + 1+ pad_e])
   
    answer = answer.replace(' ','')
    
    answer = search_paragraphs(answer,pad_s,pad_e)
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer

def search_paragraphs(answer,pad_s,pad_e):
    with open(f"./hw7_test.json","r",encoding="utf-8") as f:
        d = json.loads(f.read())
    if '[UNK]' in answer:
        for paragraph in d['paragraphs']:
            n_pattern = answer.replace('[UNK]','[\S]')
            res = re.search(r''+n_pattern,paragraph)
            if res:
                answer = res.group(0)[pad_s:] if not pad_e else res.group(0)[pad_s:-pad_e]
                #print(answer)
                return answer
    answer = answer[pad_s:] if not pad_e else answer[pad_s:-pad_e]
    #print(answer)
    return answer
def average_checkpoints(opt, save_mode):
    number = opt.cv_number if save_mode == 'ensemble' else opt.avg_number
    if not save_mode == 'ensemble':
        output_model_path = f"{opt.model_save_dir}/{save_mode}_{number}_checkpoint_{opt.current_cv_number}.pt"
    else:
        output_model_path = f"{opt.model_save_dir}/{save_mode}_{number}_checkpoint.pt"
    ckpt_folder = [folder for folder in os.listdir(opt.model_save_dir) if os.path.isdir(f"{opt.model_save_dir}/{folder}")]
    ckpts = [ckpt for ckpt in os.listdir(opt.model_save_dir) if not os.path.isdir(f"{opt.model_save_dir}/{ckpt}") and 'None' not in ckpt]
    if save_mode == 'avg_last':
        target_folder = sorted(ckpt_folder, key = lambda c : int(c.split('-')[-1]))[-number:]
        for folder in target_folder:
            src = f"{opt.model_save_dir}/{folder}/pytorch_model.bin"
            tgt = f"{opt.model_save_dir}/{folder.replace('-','_')}"
            shutil.copy(src,tgt)
    elif save_mode == 'avg_best':
        with open(f"{opt.model_save_dir}/ckpt.txt",'r') as f:
            score_record = f.read()
            score_record = list(set(score_record.split('\n')[:-1]))
            score_list = [{"folder":c.split(',')[0],"score":int(c.split(',')[1])} for c in score_record]
        target_dict = sorted(score_list, key=lambda c:c['score'])[-number:]
        target_folder = [ckpt['folder'] for ckpt in target_dict]
        for folder in target_folder:
            src = f"{opt.model_save_dir}/{folder}/pytorch_model.bin"
            tgt = f"{opt.model_save_dir}/{folder.replace('-','_')}.pt"
            shutil.copy(src,tgt)
    elif save_mode == 'ensemble':
        for ckpt in ckpts:
            src = f"{opt.model_save_dir}/{ckpt}"
            tgt = f"{opt.model_save_dir}/checkpoint{ckpt.split('checkpoint')[-1]}"
            shutil.move(src ,tgt)
    else:
        raise Exception(f'unknown save strategy : {save_mode}, Please choose your save strategy from below : avg_last , avg_best , ensemble')

    sys.argv = sys.argv + ["--inputs"] + [opt.model_save_dir] + ["--num-epoch-checkpoints"] + [str(number)] + ["--output"] + [output_model_path]
    print(sys.argv)
    main()
    ckpts = [c for c in os.listdir(opt.model_save_dir) if 'pt' in c and 'avg' not in c and 'best' not in c]
    for ckpt in ckpts:
        os.remove(f"{opt.model_save_dir}/{ckpt}")
