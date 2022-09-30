from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def get_model_and_token(opt):
    model = AutoModelForQuestionAnswering.from_pretrained(f"uer/roberta-base-chinese-extractive-qa").to(opt.device)
    tokenizer = AutoTokenizer.from_pretrained(f"uer/roberta-base-chinese-extractive-qa")
    return model, tokenizer




