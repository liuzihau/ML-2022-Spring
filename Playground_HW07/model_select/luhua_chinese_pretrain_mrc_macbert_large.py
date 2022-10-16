from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torchinfo import summary

def get_model_and_token(opt):
    model_name = "chinese_pretrain_mrc_macbert_large"
    model = AutoModelForQuestionAnswering.from_pretrained(f"luhua/{model_name}").to(opt.device)
    tokenizer = AutoTokenizer.from_pretrained(f"luhua/{model_name}")
    summary(model,input_size=(2,512),dtypes=['torch.IntTensor'])
    return model, tokenizer
if __name__ == '__main__':
    class test:
        def __init__(self):
            self.device = 'cuda'
    opt = test()
    get_model_and_token(opt)
