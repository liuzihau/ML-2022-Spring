from transformers import BertForQuestionAnswering, BertTokenizerFast
def get_model_and_token(opt):
    model = BertForQuestionAnswering.from_pretrained(opt.model_name).to(opt.device)
    tokenizer = BertTokenizerFast.from_pretrained(opt.model_name)   
    return model, tokenizer 
