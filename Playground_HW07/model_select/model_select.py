from transformers import AutoTokenizer, BertTokenizerFast, AutoModelForQuestionAnswering

def get_model_and_token(model_name, device):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    
    if model_name == 'ckiplab/bert-base-chinese-qa':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
