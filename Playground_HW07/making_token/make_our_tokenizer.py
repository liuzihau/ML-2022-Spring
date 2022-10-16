import json
from transformers import BertTokenizerFast
import tokenizers
"""
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = Whitespace()
files = [f"./hw7_{split}.json" for split in ["test", "train", "dev"]]
tokenizer.train(files, trainer)
tokenizer.save(f"./tokenizer_hw7.json")

tokenizer = Tokenizer.from_file("./tokenizer_hw7.json")
output = tokenizer.encode("士官長的頭盔上會有何裝飾")
print(output.tokens)
"""
with open(f"./hw7_test.json","r",encoding="utf-8") as f:
    data = json.loads(f.read())
vocab=[]
for paragraph in data['paragraphs']:
    vocab += paragraph
with open(f"vocab.json",'w',encoding="utf-8") as f:
    json.dump(vocab,f,indent=4,ensure_ascii=False)
bwpt = tokenizers.BertWordPieceTokenizer()
filepath = './vocab.json'
bwpt.train(files = [filepath],vocab_size=100000,min_frequency=1,limit_alphabet=100000)
bwpt.save_model("./pretrained_models/")

tokenizer = BertTokenizerFast("./pretrained_models/vocab.txt")
res = tokenizer.tokenize("a")
print(res)

