import json
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AdamW
from tqdm.auto import tqdm

def get_trainer(opt, model, train_dataset, dev_dataset, dev_questions, tokenizer):
    args = TrainingArguments(
            output_dir=opt.model_save_dir,
            seed=opt.seed,
            do_train=opt.train,
            do_eval=opt.train,
            num_train_epochs=opt.num_epoch,
            evaluation_strategy="steps",
            eval_steps=opt.logging_step,
            save_strategy="steps",
            save_steps=opt.logging_step,
            fp16=opt.fp16_training,
            per_device_train_batch_size=opt.train_batch_size,
            gradient_accumulation_steps=opt.accum_iter,
            learning_rate=opt.learning_rate,
            warmup_ratio = opt.warm_up_ratio, 
            label_smoothing_factor=opt.label_smoothing_factor
            )
    trainer = QATrainer(
              opt=opt,
              model=model,
              args=args,
              train_dataset=train_dataset,
              eval_dataset=dev_dataset,
              dev_questions=dev_questions,
              tokenizer=tokenizer
              )
    return trainer

class QATrainer(Trainer):
    def __init__(self, *args, opt=None, eval_example=None, post_process_function=None, dev_questions=None, **kwargs):
        super().__init__(*args,**kwargs)
        self.opt = opt
        self.eval_example = eval_example
        self.post_process_function = post_process_function
        self.dev_questions = dev_questions
        self.model_save_dir = opt.model_save_dir
        self.cv_number = opt.current_cv_number if hasattr(opt,'current_cv_number') else None
        self.best_dev_acc_avg = 0
        if opt.train:
            with open(f'{opt.model_save_dir}/ckpt.txt','w') as f:
                pass
    
    def evaluate(self):
        usage = 'dev'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Evaluating Dev Set ...")
        self.model.eval()
        dev_acc = 0
        dev_loader = DataLoader(self.eval_dataset, batch_size=1, shuffle=False, pin_memory=True)
        with torch.no_grad():
            for i, data in enumerate(tqdm(dev_loader)):
                sub_dataset = {'input_ids':data[0].squeeze(dim=0).to(device),
                               'token_type_ids':data[1].squeeze(dim=0).to(device),
                               'attention_mask':data[2].squeeze(dim=0).to(device),
                               'start_positions':data[3].squeeze(dim=0).to(device),
                               'end_positions':data[4].squeeze(dim=0).to(device)}
                
                output = self.model(input_ids=sub_dataset['input_ids'], token_type_ids=sub_dataset['token_type_ids'], attention_mask=sub_dataset['attention_mask'])
                # prediction is correct only if answer text exactly matches
                answer_predicted = postprocess_output_data(data, self.tokenizer, output)
                if i % 500==0:
                    print(f'[Info] answer_sample : {answer_predicted} | answer: {self.dev_questions[i]["answer_text"]}')
                dev_acc += answer_predicted == self.dev_questions[i]["answer_text"]
                dev_acc_avg = dev_acc / len(dev_loader)
            print(f"Validation | acc = {dev_acc_avg:.3f}")
        self.model.train()
        if self.opt.save_mode == 'best':
            if dev_acc_avg > self.best_dev_acc_avg:
                print("Saving Best Model ...")
                torch.save(self.model.state_dict(), f"{self.model_save_dir}/best_None_checkpoint_{self.cv_number}.pt")
                self.best_dev_acc_avg = dev_acc_avg
        elif self.opt.save_mode == 'avg_best':
            with open(f'{self.opt.model_save_dir}/ckpt.txt','a') as f:
                dev_acc_avg = 0.99999 if dev_acc_avg == 1 else dev_acc_avg
                score = f"{dev_acc_avg:.5f}".split('.')[-1] 
                f.write(f"checkpoint-{self.state.global_step},{score}\n")


    def test_model(self, opt, test_questions, test_loader):
        usage = 'test'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Evaluating Test Set ...")
        result = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader)):
                sub_dataset = {'input_ids':data[0].squeeze(dim=0).to(device),
                               'token_type_ids':data[1].squeeze(dim=0).to(device),
                               'attention_mask':data[2].squeeze(dim=0).to(device)}

                output = self.model(input_ids=sub_dataset['input_ids'], token_type_ids=sub_dataset['token_type_ids'], attention_mask=sub_dataset['attention_mask'])
                answer = postprocess_output_data(data, self.tokenizer, output) 
                result.append(answer)

        result_file = "result.csv"
        with open(result_file, 'w') as f:	
            f.write("ID,Answer\n")
            for i, test_question in enumerate(test_questions):
                # Replace commas in answers with empty strings (since csv is separated by comma)
                # Answers in kaggle are processed in the same way
                f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

        print(f"Completed! Result is in {result_file}")


def postprocess_output_data(data, tokenizer, output):
    n_best = 20
    max_answer_length = 30
    np_data = data[0][0].cpu().numpy() # first list : input ids second array : batch
    start_logits = output.start_logits.cpu().numpy()
    end_logits = output.end_logits.cpu().numpy()
    answers = []
    
    num_of_windows = data[0].shape[1]

    for k in range(num_of_windows):
        SEP = np.sort(np.where(np_data[k] == 102))[0]
        start_indexes = np.argsort(start_logits[k])[::-1][:n_best]
        end_indexes = np.argsort(end_logits[k])[::-1][:n_best]
        for start_index in start_indexes:
            for end_index in end_indexes:
                # exclude some nonsence situation
                if start_index <= SEP[0] or end_index >= SEP[1]:
                    continue
                if end_index <= start_index:
                    continue
                if end_index - start_index + 1 > max_answer_length:
                    continue

                # cook answer candidate
                text = tokenizer.decode(np_data[k][start_index : end_index + 1])
                text = text.replace(' ','')
                score = start_logits[k][start_index] + end_logits[k][end_index]
                
                # pad text in [UNK] case
                pad_s = pad_e = 0
                unk = '[UNK]' in text
                if unk:
                    text = text.replace('##','') # remove sub-word token
                    for i in range(1,6):
                        char = tokenizer.decode([np_data[k][start_index-i]])
                        if char in ['[UNK]','[SEP]']:
                            break
                        char = char.replace('##','') # remove sub-word token
                        text = char + text
                        pad_s += len(char)
                    for i in range(1,6):
                        char = tokenizer.decode([np_data[k][end_index+i]])
                        if char in ['[UNK]','[SEP]']:
                            break
                        char = char.replace('##','') # remove sub-word token
                        text = text + char
                        pad_e += len(char)
                
                # add answer candidate to answer list
                answers.append({
                        'text': text,
                        'score': score,
                        'unk': unk,
                        'pad_s': pad_s,
                        'pad_e' :pad_e
                        })
    
    if answers:
        answer = max(answers, key = lambda c : c['score'])
        if answer['unk']:
            answer_text = search_paragraphs(answer, data[-1][0])
        else:
            answer_text = answer['text']
    else:
        answer_text = ''

    return answer_text

def search_paragraphs(answer, paragraph):
    unk_index = answer['text'].index('[UNK]')
    print(f"[Info] Find [UNK] in answer : {answer['text']}")
    n_pattern = answer['text'].replace('[UNK]','[\S]{1,8}') if (unk_index > 0 or unk_index < len(answer['text'])-1) else answer['text'].replace('[UNK]','[\S]') 
    res = re.search(r''+n_pattern, paragraph)
    if res:
        text = res.group(0)[answer['pad_s']:] if answer['pad_e'] == 0 else res.group(0)[answer['pad_s']:-answer['pad_e']]
        print(f"[Info] after replace : {text}")
        if answer['pad_e'] == 0:
            print(f"[Info] origin answer : {answer['text'][answer['pad_s']:]}")
        else:
            print(f"[Info] origin answer : {answer['text'][answer['pad_s']:-answer['pad_e']]}")
        return text
    text = answer['text'][answer['pad_s']:] if answer['pad_e'] == 0 else answer['text'][answer['pad_s']:-answer['pad_e']]
    return text
