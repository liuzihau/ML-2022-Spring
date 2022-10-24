import os
import sys
import datetime
import shutil
import json
import yaml
import random
import torch
from torch.utils.data import DataLoader 
from tqdm.auto import tqdm

from model_select.model_select import get_model_and_token
from utils import get_config, same_seeds, read_data, tokenize_data, average_checkpoints
from w_trainer.dataset import QA_Dataset
from w_trainer.trainer import get_trainer
from fairseq.average_checkpoints import main 

opt = get_config("./w_trainer/config.yaml")
opt.model_save_dir = f"{opt.model_save_dir}_{opt.model_name.replace('/','_')}_latest"
opt.device = "cuda" if torch.cuda.is_available() else "cpu"
# Fix random seed for reproducibility
same_seeds(opt.seed)

print(f"[Info] Load model and tokenizer...")
model, tokenizer = get_model_and_token(opt.model_name, opt.device)
print(f"[Info] Load success!")
trainer = None

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
            train_set = QA_Dataset(opt, "train", train_questions, paragraphs, train_questions_tokenized, paragraphs_tokenized)
            dev_questions_tokenized, paragraphs_tokenized = tokenize_data(tokenizer, dev_questions, paragraphs)
            dev_set = QA_Dataset(opt, "dev", dev_questions, paragraphs, dev_questions_tokenized, paragraphs_tokenized)
            
            print(f"[Info] Load trainer...")
            opt.current_cv_number = k
            trainer = get_trainer(opt, model, train_set, dev_set,dev_questions, tokenizer)
            print(f"[Info] Load success!")
            trainer.train()
            if opt.save_mode == 'avg_best' or 'avg_last':
                average_checkpoints(opt, opt.save_mode)

            model, tokenizer = get_model_and_token(opt.model_name, opt.device)
        
        average_checkpoints(opt, 'ensemble')
        checkpoint = torch.load(output_model_path)
        model.load_state_dict(checkpoint['model'])
    else:
        train_questions, train_paragraphs = read_data(opt.train_data_name)
        dev_questions, dev_paragraphs = read_data(opt.dev_data_name)
        
        train_questions_tokenized, train_paragraphs_tokenized = tokenize_data(tokenizer,train_questions,train_paragraphs)
        train_set = QA_Dataset(opt, "train", train_questions,train_paragraphs, train_questions_tokenized, train_paragraphs_tokenized)
        dev_questions_tokenized, dev_paragraphs_tokenized = tokenize_data(tokenizer,dev_questions,dev_paragraphs)
        dev_set = QA_Dataset(opt, "dev", dev_questions,dev_paragraphs, dev_questions_tokenized, dev_paragraphs_tokenized)
    
        print(f"[Info] Load trainer...")
        opt.current_cv_number = None
        trainer = get_trainer(opt, model, train_set, dev_set, dev_questions, tokenizer)
        print(f"[Info] Load success!")
        trainer.train()


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
                if os.path.isdir(load_path):
                    continue
                break
    print(f"[Info] use checkpoint : {load_path}")
    checkpoint = torch.load(load_path)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

test_questions, test_paragraphs = read_data(opt.test_data_name)

test_questions_tokenized, test_paragraphs_tokenized = tokenize_data(tokenizer,test_questions,test_paragraphs)
test_set = QA_Dataset(opt, "test", test_questions, test_paragraphs, test_questions_tokenized, test_paragraphs_tokenized)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

if not trainer:
    train_questions, train_paragraphs = read_data(opt.train_data_name)
    train_questions_tokenized, train_paragraphs_tokenized = tokenize_data(tokenizer,train_questions,train_paragraphs)
    train_set = QA_Dataset(opt, "train", train_questions, train_paragraphs, train_questions_tokenized, train_paragraphs_tokenized)
    dev_questions, dev_paragraphs = read_data(opt.dev_data_name)
    dev_questions_tokenized, dev_paragraphs_tokenized = tokenize_data(tokenizer,dev_questions,dev_paragraphs)
    dev_set = QA_Dataset(opt, "dev", dev_questions, dev_paragraphs, dev_questions_tokenized, dev_paragraphs_tokenized)

    print(f"[Info] Load trainer...")
    trainer = get_trainer(opt, model, train_set, dev_set, dev_questions, tokenizer)
    print(f"[Info] Load success!")
trainer.test_model(opt, test_questions, test_loader)
