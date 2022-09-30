from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def get_model_and_token(opt):
    model_name = "chinese_pretrain_mrc_roberta_wwm_ext_large" # "chinese_pretrain_mrc_macbert_large"
    model = AutoModelForQuestionAnswering.from_pretrained(f"luhua/{model_name}").to(opt.device)
    tokenizer = AutoTokenizer.from_pretrained(f"luhua/{model_name}")
    return model, tokenizer




