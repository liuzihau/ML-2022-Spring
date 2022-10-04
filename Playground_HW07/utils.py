import random
import numpy as np
import torch
import yaml
import json
import re
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

