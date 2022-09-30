from transformers import (
  BertTokenizerFast,
  AutoModel,
  AutoModelForQuestionAnswering
)
def get_model_and_token(opt):
    # model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-qa').to(opt.device)
    model = AutoModelForQuestionAnswering.from_pretrained('ckiplab/bert-base-chinese-qa').to(opt.device)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    return model, tokenizer
