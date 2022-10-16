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

from utils import get_config, same_seeds,read_data, tokenize_data
from w_trainer.dataset import QA_Dataset
from w_trainer.trainer import get_trainer
from fairseq.average_checkpoints import main 

opt = get_config("./w_trainer/config.yaml")

save_path =f"{opt.model_save_dir}_{opt.model_name}_latest"
opt.model_save_dir = save_path

opt.device = "cuda" if torch.cuda.is_available() else "cpu"
# Fix random seed for reproducibility
same_seeds(opt.seed)

if opt.model_name == "bert-base-chinese":
    from model_select.bert_base_chinese import get_model_and_token
elif opt.model_name == "ckiplab-bert-base-chinese-qa":
    from model_select.ckiplab_bert_base_chinese_qa import get_model_and_token
elif opt.model_name == "hfl_chinese_roberta_wwm_ext_large":
    from model_select.hfl_chinese_roberta_wwm_ext_large import get_model_and_token
elif opt.model_name == "luhua_chinese_pretrain_mrc_roberta_wwm_ext_large":
    from model_select.luhua_chinese_pretrain_mrc_roberta_wwm_ext_large import get_model_and_token
elif opt.model_name == "luhua_chinese_pretrain_mrc_macbert_large":
    from model_select.luhua_chinese_pretrain_mrc_macbert_large import get_model_and_token
elif opt.model_name == "uer_roberta-base-chinese-extractive-qa":
    from model_select.uer_roberta_base_chinese_extractive_qa import get_model_and_token

print(f"[Info] Load model and tokenizer...")
model, tokenizer = get_model_and_token(opt)
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
        if opt.tw_to_s:
            from opencc import OpenCC
            cc = OpenCC('tw2s')
            for question in questions1:
                question['question_text'] = cc.convert(question['question_text'])
                question['answer_text'] = cc.convert(question['answer_text'])
            for question in questions2:
                question['question_text'] = cc.convert(question['question_text'])
                question['answer_text'] = cc.convert(question['answer_text'])
            paragraphs1 = [cc.convert(paragraph) for paragraph in paragraphs1]
            paragraphs2 = [cc.convert(paragraph) for paragraph in paragraphs2]

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
            dev_questions_tokenized, paragraphs_tokenized = tokenize_data(tokenizer, dev_questions, paragraphs)
            dev_set = QA_Dataset(opt, "dev", dev_questions, dev_questions_tokenized, paragraphs_tokenized)
            
            print(f"[Info] Load trainer...")
            opt.current_cv_number = k
            trainer = get_trainer(opt, model, train_set, dev_set,dev_questions, tokenizer)
            print(f"[Info] Load success!")
            trainer.train()

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
        dev_questions, dev_paragraphs = read_data(opt.dev_data_name)
        if opt.tw_to_s:
            from opencc import OpenCC
            cc = OpenCC('tw2s')
            for question in train_questions:
                question['question_text'] = cc.convert(question['question_text'])
                question['answer_text'] = cc.convert(question['answer_text'])
            train_paragraphs = [cc.convert(paragraph) for paragraph in train_paragraphs]
            for question in dev_questions:
                question['question_text'] = cc.convert(question['question_text'])
                question['answer_text'] = cc.convert(question['answer_text'])
            dev_paragraphs = [cc.convert(paragraph) for paragraph in dev_paragraphs]
        train_questions_tokenized, train_paragraphs_tokenized = tokenize_data(tokenizer,train_questions,train_paragraphs)
        train_set = QA_Dataset(opt, "train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
        dev_questions_tokenized, dev_paragraphs_tokenized = tokenize_data(tokenizer,dev_questions,dev_paragraphs)
        dev_set = QA_Dataset(opt, "dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
    
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
if opt.tw_to_s:
    from opencc import OpenCC
    cc = OpenCC('tw2s')
    for question in test_questions:
        question['question_text'] = cc.convert(question['question_text'])
    test_paragraphs = [cc.convert(paragraph) for paragraph in test_paragraphs]

test_questions_tokenized, test_paragraphs_tokenized = tokenize_data(tokenizer,test_questions,test_paragraphs)
test_set = QA_Dataset(opt, "test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

if not trainer:
    train_questions, train_paragraphs = read_data(opt.train_data_name)
    train_questions_tokenized, train_paragraphs_tokenized = tokenize_data(tokenizer,train_questions,train_paragraphs)
    train_set = QA_Dataset(opt, "train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
    dev_questions, dev_paragraphs = read_data(opt.dev_data_name)
    dev_questions_tokenized, dev_paragraphs_tokenized = tokenize_data(tokenizer,dev_questions,dev_paragraphs)
    dev_set = QA_Dataset(opt, "dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)

    print(f"[Info] Load trainer...")
    trainer = get_trainer(opt, model, train_set, dev_set, dev_questions, tokenizer)
    print(f"[Info] Load success!")
trainer.test_model(opt, test_questions, test_loader)
