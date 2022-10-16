import os
import random
import torch
from torch.utils.data import Dataset

class QA_Dataset(Dataset):
    def __init__(self, opt, split, questions, tokenized_questions, tokenized_paragraphs):
        super().__init__()
        self.split = split
        self.opt = opt
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = opt.max_question_len
        self.max_paragraph_len = opt.max_paragraph_len
        self.sent_length = self.max_question_len + self.max_paragraph_len + 3 # [101]+Q+[102]+P+[102]
        self.doc_stride = int(opt.doc_stride * self.max_paragraph_len)
        self.pick_method = opt.pick_method if 'pick_method' in opt else 'random'
        self.greedy_data = []
        if self.pick_method == 'greedy' and self.split == 'train':
            for idx in range(len(self.questions)):
                question = self.tokenized_questions[idx]
                paragraph = self.tokenized_paragraphs[self.questions[idx]["paragraph_id"]]
                answer_start = self.questions[idx]["answer_start"]
                answer_end = self.questions[idx]["answer_end"]
                answer_start_tokenized = paragraph.char_to_token(answer_start)
                answer_end_tokenized = paragraph.char_to_token(answer_end)
                for i in range(0, len(paragraph), self.doc_stride):

                    # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                    question_sent_truc = [101] + question.ids[:self.max_question_len] + [102]
                    paragraph_sent_truc = paragraph.ids[i : i + self.max_paragraph_len] + [102]

                    # Pad sequence and obtain inputs to model
                    sentence, segmentation, attention_mask = self.make_sentence(question_sent_truc, paragraph_sent_truc)
                    ans_start_shift, ans_end_shift = self.shift_answer(answer_start_tokenized, answer_end_tokenized, i, len(question_sent_truc))

                    self.greedy_data.append({
                        'input_ids':torch.tensor(sentence),
                        'token_type_ids':torch.tensor(segmentation),
                        'attention_mask':torch.tensor(attention_mask),
                        'start_positions':torch.tensor(ans_start_shift),
                        'end_positions':torch.tensor(ans_end_shift)})
                    
 

    def __getitem__(self,idx):
        if self.pick_method == 'greedy' and self.split=='train':
            return self.greedy_data[idx]
        """
        [CLS] question (truncate at max question length) [SEP] paragraph (truncate at max paragraph length) [SEP] 
        [CLS] = [101] [SEP] = [102]
        """
        question = self.tokenized_questions[idx]
        paragraph = self.tokenized_paragraphs[self.questions[idx]["paragraph_id"]]
        if self.split in ['train', 'dev']:
            answer_start = self.questions[idx]["answer_start"]
            answer_end = self.questions[idx]["answer_end"]
            answer_start_tokenized = paragraph.char_to_token(answer_start)
            answer_end_tokenized = paragraph.char_to_token(answer_end)
        if self.split == 'train':

            # random chioce paragraph start point
            if self.pick_method == 'random':
                if random.uniform(0, 1) < self.opt.random_pick_ratio:
                    paragraph_start, paragraph_end = self.pick_paragraph_randomly(question, paragraph, answer_start_tokenized, answer_end_tokenized)
                else:
                    paragraph_start, paragraph_end = self.pick_paragraph_w_answer(question, paragraph, answer_start_tokenized, answer_end_tokenized)

                question_sent_truc = [101] + question.ids[:self.max_question_len] + [102]
                paragraph_sent_truc = paragraph.ids[paragraph_start : paragraph_end] + [102]

                sentence, segmentation, attention_mask = self.make_sentence(question_sent_truc, paragraph_sent_truc)

                # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window
                ans_start_shift, ans_end_shift = self.shift_answer(answer_start_tokenized, answer_end_tokenized, paragraph_start, len(question_sent_truc))
            
            return {'input_ids':torch.tensor(sentence),
                    'token_type_ids':torch.tensor(segmentation),
                    'attention_mask':torch.tensor(attention_mask),
                    'start_positions':torch.tensor(ans_start_shift),
                    'end_positions':torch.tensor(ans_end_shift)}

        else: # dev or test
           
            sentence_list, segmentation_list, attention_mask_list = [], [], []
            ans_start_list, ans_end_list = [], []
   
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            for i in range(0, len(paragraph), self.doc_stride):
                
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                question_sent_truc = [101] + question.ids[:self.max_question_len] + [102]
                paragraph_sent_truc = paragraph.ids[i : i + self.max_paragraph_len] + [102]

                # Pad sequence and obtain inputs to model
                sentence, segmentation, attention_mask = self.make_sentence(question_sent_truc, paragraph_sent_truc)
                sentence_list.append(sentence)
                segmentation_list.append(segmentation)
                attention_mask_list.append(attention_mask)

                if self.split == 'dev':
                    ans_start_shift, ans_end_shift = self.shift_answer(answer_start_tokenized, answer_end_tokenized, i, len(question_sent_truc))
                    ans_start_list.append(ans_start_shift)
                    ans_end_list.append(ans_end_shift)
            
            return torch.tensor(sentence_list), torch.tensor(segmentation_list), torch.tensor(attention_mask_list), torch.tensor(ans_start_list), torch.tensor(ans_end_list)

    
    def __len__(self):
        if self.greedy_data:
            return len(self.greedy_data)
        return len(self.questions)

    def pick_paragraph_randomly(self, question, paragraph, answer_start, answer_end):
        paragraph_start = random.randint(0, len(paragraph.ids))
        paragraph_end = paragraph_start + self.max_paragraph_len
        
        return paragraph_start, paragraph_end

    def pick_paragraph_w_answer(self, question, paragraph, answer_start, answer_end):
        # Answer could appear anywhere in paragrhph[pad:-pad]
        pad = int(self.max_paragraph_len * self.opt.answer_pad)
        rand_point = random.choice(range(pad, self.max_paragraph_len + answer_end - answer_start - pad))
        paragraph_start = max(0, answer_start - rand_point)
        paragraph_end = paragraph_start + self.max_paragraph_len
        
        return paragraph_start, paragraph_end

    def make_sentence(self, question, paragraph):
        pad_length = self.sent_length - len(question) - len(paragraph)
        sentence = question + paragraph + [0] * pad_length
        segmentation = [0] * len(question) + [1] * len(paragraph) + [0] * pad_length
        attention_mask = [1] * (len(question) + len(paragraph)) + [0] * pad_length
        return sentence, segmentation, attention_mask

    def shift_answer(self, answer_start, answer_end, paragraph_start, question_length):
        if answer_start - paragraph_start < 0 or answer_end - paragraph_start > self.max_paragraph_len:
            answer_start = answer_end = 0
        else:
            answer_start += question_length - paragraph_start
            answer_end += question_length - paragraph_start
        return answer_start, answer_end


if __name__ == "__main__":
    from transformers import BertForQuestionAnswering, BertTokenizerFast
    import yaml
    import json
    
    def get_token(opt):
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
        return tokenizer
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    def read_data(file):
        with open(file, 'r', encoding="utf-8") as reader:
            data = json.load(reader)
        return data["questions"], data["paragraphs"]
    def tokenize_data(tokenizer, questions, paragraphs):
        questions_tokenized = tokenizer([question["question_text"] for question in questions], add_special_tokens=False)
        paragraphs_tokenized = tokenizer(paragraphs, add_special_tokens=False)
        return questions_tokenized, paragraphs_tokenized

    def get_config(file_path):
        with open(file_path, 'r', encoding="utf-8") as stream:
            opt = yaml.safe_load(stream)
        opt = AttrDict(opt)
        return opt

    opt = get_config("./config/config.yaml")
    print(f"[Info] Load model and tokenizer...")
    tokenizer = get_token(opt)
    print(f"[Info] Load success!")
    train_questions, train_paragraphs = read_data(opt.train_data_name)
    train_questions_tokenized, train_paragraphs_tokenized = tokenize_data(tokenizer,train_questions,train_paragraphs)
    train_set = QA_Dataset('test', opt, train_questions, train_questions_tokenized, train_paragraphs_tokenized)
    for i in range(5):
        target = train_set[1]
        print('='*80)
        print('sentence')
        for i in range(target[0].shape[0]):
            print(tokenizer.decode(target[0][i]))
        # print('answer')
        # print(tokenizer.decode(target[0][target[3]:target[4]+1]))
        print('#'*80)
