import os
import sys
import datetime
import shutil
import json
import yaml
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset 
from tqdm.auto import tqdm

from utils import get_config, same_seeds,read_data, tokenize_data
from no_trainer.dataset import QA_Dataset
from no_trainer.train_script import train_model, test_model
from fairseq.average_checkpoints import main 
from transformers import BertTokenizerFast

opt = get_config("./config/config.yaml")

save_path =f"{opt.model_save_dir}_{opt.model_name}_latest"
"""
opt.model_save_dir =f"{opt.model_save_dir}_{opt.model_name}_{datetime.date.today()}"
if os.path.exists(save_path):
    f = get_config(f"{save_path}/config.yaml")
    shutil.move(save_path, f.model_save_dir)
os.mkdir(save_path)
with open(f'{save_path}/config.yaml', 'w') as f:
    yaml.dump(opt, f)
"""
opt.model_save_dir = save_path

opt.device = "cuda" if torch.cuda.is_available() else "cpu"
# Fix random seed for reproducibility
same_seeds(opt.seed)

# Change "fp16_training" to True to support automatic mixed precision training (fp16)	
# Documentation for the toolkit:  https://huggingface.co/docs/accelerate/
if opt.fp16_training:
    from accelerate import Accelerator
    opt.accelerator = Accelerator(fp16=True)
    opt.device = opt.accelerator.device

if opt.model_name == "bert-base-chinese":
    from model_select.bert_base_chinese import get_model_and_token
elif opt.model_name == "ckiplab-bert-base-chinese-qa":
    from model_select.ckiplab_bert_base_chinese_qa import get_model_and_token
elif opt.model_name == "luhua_chinese_pretrain_mrc_roberta_wwm_ext_large":
    from model_select.luhua_chinese_pretrain_mrc_roberta_wwm_ext_large import get_model_and_token
elif opt.model_name == "uer_roberta-base-chinese-extractive-qa":
    from model_select.uer_roberta_base_chinese_extractive_qa import get_model_and_token

print(f"[Info] Load model and tokenizer...")
model, tokenizer = get_model_and_token(opt)

print(f"[Info] Load success!")

# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair

if opt.train:
    if opt.ensemble:
        
        questions1, paragraphs1 = read_data(opt.train_data_name)
        question_start_index = len(questions1)
        paragraph_start_index = len(paragraphs1)
        questions2, paragraphs2 = read_data(opt.dev_data_name)
        for q in questions2:
            q["id"] += question_start_index
            q["paragraph_id"] += paragraph_start_index
        questions = questions1 + questions2
        paragraphs = paragraphs1 + paragraphs2
        data_length = len(questions)
        dev_length = data_length // opt.cv_number
    
        for k in range(opt.cv_number):
            if k == 0:
                train_questions = questions[(k+1)*dev_length:]
            elif k == opt.cv_number-1:
                train_questions = questions[:k*dev_length]
            else:
                train_questions = questions[:k*dev_length] + questions[(k+1)*dev_length:]
            dev_questions = questions[k*dev_length:(k+1)*dev_length]
        
            train_questions_tokenized, paragraphs_tokenized = tokenize_data(tokenizer,train_questions, paragraphs)
            train_set = QA_Dataset(opt, "train", train_questions, train_questions_tokenized, paragraphs_tokenized)
            train_loader = DataLoader(train_set, batch_size=opt.train_batch_size, shuffle=True, pin_memory=True)
            dev_questions_tokenized, paragraphs_tokenized = tokenize_data(tokenizer, dev_questions, paragraphs)
            dev_set = QA_Dataset(opt, "dev", dev_questions, dev_questions_tokenized, paragraphs_tokenized)
            dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
    
            train_model(model, opt, tokenizer, train_loader, dev_questions, dev_loader, k)
        
            model, tokenizer = get_model_and_token(opt)
        
        # need do average
        output_model_path = f"{opt.model_save_dir}/avg_last_{opt.cv_number}_checkpoint.pt"
        sys.argv = sys.argv + ["--inputs"] + [opt.model_save_dir] + ["--num-epoch-checkpoints"] + [str(opt.cv_number)] + ["--output"] + [output_model_path]
        print(sys.argv)
        main()
        checkpoint = torch.load(output_model_path)
        model.load_state_dict(checkpoint['model'])
    else:
        train_questions, train_paragraphs = read_data(opt.train_data_name)
        train_questions_tokenized, train_paragraphs_tokenized = tokenize_data(tokenizer,train_questions,train_paragraphs)
        train_set = QA_Dataset(opt, "train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
        train_loader = DataLoader(train_set, batch_size=opt.train_batch_size, shuffle=True, pin_memory=True)
        dev_questions, dev_paragraphs = read_data(opt.dev_data_name)
        dev_questions_tokenized, dev_paragraphs_tokenized = tokenize_data(tokenizer,dev_questions,dev_paragraphs)
        dev_set = QA_Dataset(opt, "dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
        dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
    
        train_model(model, opt, tokenizer, train_loader, dev_questions, dev_loader)


if opt.use_finetune_model:
    file_list = os.listdir(opt.model_save_dir)
    load_path = None
    for f in file_list:
        if 'avg' in f:
            load_path = f"{opt.model_save_dir}/{f}"
            break
    if not load_path:
        for f in file_list:
            if 'checkpoint' in f:
                load_path = f"{opt.model_save_dir}/{f}"
    checkpoint = torch.load(load_path)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

test_questions, test_paragraphs = read_data(opt.test_data_name)
test_questions_tokenized, test_paragraphs_tokenized = tokenize_data(tokenizer,test_questions,test_paragraphs)
test_set = QA_Dataset(opt, "test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)
test_model(model, opt, tokenizer, test_questions, test_loader)

