import json
import re
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
        self.eval_example = eval_example
        self.post_process_function = post_process_function
        self.dev_questions = dev_questions
        self.model_save_dir = opt.model_save_dir
        self.cv_number = opt.current_cv_number if hasattr(opt,'current_cv_number') else None
        self.best_dev_acc_avg = 0
    
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
                answer_predicted = postprocess_output_data(usage, i, data, self.tokenizer, output)
                if i % 500==0:
                    print(f'[Info] answer_sample : {answer_predicted} | answer: {self.dev_questions[i]["answer_text"]}')
                dev_acc += answer_predicted == self.dev_questions[i]["answer_text"]
                dev_acc_avg = dev_acc / len(dev_loader)
            print(f"Validation | acc = {dev_acc_avg:.3f}")
        self.model.train()
        if dev_acc_avg > self.best_dev_acc_avg:
            print("Saving Best Model ...")
            torch.save(self.model.state_dict(), f"{self.model_save_dir}/checkpoint_{self.cv_number}.pt")
            self.best_dev_acc_avg = dev_acc_avg


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
                answer = postprocess_output_data(usage, i, data, self.tokenizer, output) 
                if opt.tw_to_s:
                    from opencc import OpenCC
                    cc = OpenCC('s2tw')
                    answer = cc.convert(answer)
                result.append(answer)

        result_file = "result.csv"
        with open(result_file, 'w') as f:	
            f.write("ID,Answer\n")
            for i, test_question in enumerate(test_questions):
                # Replace commas in answers with empty strings (since csv is separated by comma)
                # Answers in kaggle are processed in the same way
                f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

        print(f"Completed! Result is in {result_file}")


def postprocess_output_data(usage, index, data, tokenizer, output):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing
    # Hint: Open your prediction file to see what is wrong
    answer = ''
    answer_bk = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]

    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        # start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        # end_prob, end_index = torch.max(output.end_logits[k], dim=0)

        start_prob1, start_index1 = torch.max(output.start_logits[k][:-1], dim=0)
        end_prob2, end_index2 = torch.max(output.end_logits[k], dim=0)
        if start_index1 == 0 or end_index2 == 0 :
            continue

        end_prob1, end_index1 = torch.max(output.end_logits[k][start_index1+1:], dim=0)
        end_index1 = end_index1 + start_index1 + 1

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
            
            if 100 in  data[0][0][k][start_index : end_index + 1]: 
                pad_s = 7
                pad_e = 3
                target = data[0][0][k][start_index-pad_s : start_index]
                if 102 in target:
                    para_start = (target == 102).nonzero(as_tuple=True)[0].to("cuda")
                    pad_s = pad_s - para_start - 1
                    pad_e = 3         
            else:
                pad_s = 0
                pad_e = 0
            answer = tokenizer.decode(data[0][0][k][start_index-pad_s : end_index + 1+ pad_e])
            answer_bk = tokenizer.decode(data[0][0][k][start_index:end_index + 1])
    answer = answer.replace(' ','')
    answer_bk = answer_bk.replace(' ','')
    if answer: 
        answer = search_paragraphs(usage, index, answer, answer_bk, pad_s, pad_e)
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer

def search_paragraphs(usage, index, answer, answer_bk, pad_s, pad_e):
    path = "./data/hw7_dev.json" if usage == 'dev' else "./data/hw7_test.json"
    with open(path,"r",encoding="utf-8") as f:
        d = json.loads(f.read())
    if '[UNK]' in answer:
        unk_index = answer.index('[UNK]')
        print(f"[Info] Find [UNK] in answer : {answer}")
        paragraph = d['paragraphs'][d['questions'][index]['paragraph_id']]
        n_pattern = answer.replace('[UNK]','[\S]{1,8}') if unk_index > 0 else answer.replace('[UNK]','[\S]') 
        print(n_pattern)
        res = re.search(r''+n_pattern,paragraph)
        print(res)
        if res:
            answer = res.group(0)[pad_s:] if not pad_e else res.group(0)[pad_s:-pad_e]
            print(f"[Info] after replace : {answer}")
            print(f"[Info] origin answer : {answer_bk}")
            return answer
        print(f"[Info] not found replace candidate in {index} paragraph")
    answer = answer[pad_s:] if not pad_e else answer[pad_s:-pad_e]
    return answer
