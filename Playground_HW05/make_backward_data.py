import time
from pathlib import Path

import subprocess
import sentencepiece as spm
from make_data import clean_corpus

data_dir = './DATA/rawdata'
mono_dataset_name = 'mono'
mono_prefix = Path(data_dir).absolute() / mono_dataset_name
mono_prefix.mkdir(parents=True, exist_ok=True)

urls = (
    "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/ted_zh_corpus.deduped.gz",
)
file_names = (
    'ted_zh_corpus.deduped.gz',
)

for u, f in zip(urls, file_names):
    path = mono_prefix / f
    if not path.exists():
        subprocess.Popen(f"wget {u} -O {path}", shell=True)
        for i in range(20):
            print(f'Download data... wait {20 - i} sec')
            time.sleep(1)
    else:
        print(f'{f} is exist, skip downloading')
    if path.suffix == ".tgz":
        subprocess.Popen(f"tar -xvf {path} -C {mono_prefix}", shell=True)
        for i in range(10):
            print(f'extract data... wait {15 - i} sec')
            time.sleep(1)
    elif path.suffix == ".zip":
        subprocess.Popen(f"unzip -o {path} -d {mono_prefix}", shell=True)
        for i in range(10):
            print(f'extract data... wait {15 - i} sec')
            time.sleep(1)
    elif path.suffix == ".gz":
        subprocess.Popen(f"gzip -fkd {path}", shell=True)
        for i in range(10):
            print(f'extract data... wait {10 - i} sec')
            time.sleep(1)



# mono as backward test data
subprocess.Popen(f"mv {mono_prefix / 'ted_zh_corpus.deduped'} {mono_prefix / 'mono.zh'}", shell=True)
with open(f'{mono_prefix}/mono.zh', 'r') as f1:
    with open(f'{mono_prefix}/mono.en', 'w') as f2:
        flag = True
        while flag:
            if f1.readline():
                f2.writelines('.\n')
            else:
                flag = False
time.sleep(1)
src_lang = 'zh'
tgt_lang = 'en'

# TODO: clean corpus
"""
    remove sentences that are too long or too short
    unify punctuation

hint: you can use clean_s() defined above to do this
"""
mono_prefix = Path(data_dir).absolute() / mono_dataset_name / mono_dataset_name
clean_corpus(mono_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)

# TODO: Subword Units
"""
Use the spm model of the backward model to tokenize the data into subword units

hint: spm model is located at DATA/raw-data/[dataset]/spm[vocab_num].model
"""
vocab_size = 8000
dataset_name = 'ted2020'
prefix = Path(data_dir).absolute() / dataset_name

spm_model = spm.SentencePieceProcessor(model_file=str(prefix / f'spm{vocab_size}.model'))
in_tag = {
    'test': 'mono.clean',
}
mono_prefix = Path(data_dir).absolute() / mono_dataset_name

for split in ['test']:
    for lang in [src_lang, tgt_lang]:
        out_path = mono_prefix / f'{in_tag[split]}.tok.{lang}'
        if out_path.exists():
            print(f"{out_path} exists. skipping spm_encode.")
        else:
            with open(mono_prefix / f'{in_tag[split]}.tok.{lang}', 'w') as out_f:
                with open(mono_prefix / f'{in_tag[split]}.{lang}', 'r') as in_f:
                    for line in in_f:
                        line = line.strip()
                        tok = spm_model.encode(line, out_type=str)
                        print(' '.join(tok), file=out_f)

# Binarize
"""
use fairseq to binarize data
"""
binpath = Path('../DATA/data-bin', 'ted2020_backward')
src_dict_file = '../DATA/data-bin/ted2020/dict.zh.txt'
tgt_dict_file = '../DATA/data-bin/ted2020/dict.en.txt'
monopref = str(mono_prefix / "mono.clean.tok")  # whatever filepath you get after applying subword tokenization
if binpath.exists():
    print(binpath, "exists, will not overwrite!")
else:
    subprocess.Popen(
        f"python -m fairseq_cli.preprocess --source-lang 'zh' --target-lang 'en' --testpref {monopref}  --destdir {binpath} --srcdict {src_dict_file} --tgtdict {tgt_dict_file} --workers 2",
        cwd="fairseq", shell=True)
    subprocess.Popen(
        f"python -m fairseq_cli.preprocess --source-lang {src_lang} --target-lang {tgt_lang} --trainpref {prefix / 'train'} --validpref {prefix / 'valid'} --destdir {binpath} --joined-dictionary --workers 2",
        cwd="fairseq", shell=True)

# TODO: Generate synthetic data with backward model
'''
Add binarized monolingual data to the original data directory, and name it with "split_name"

ex. ./DATA/data-bin/ted2020/[split_name].zh-en.["en", "zh"].["bin", "idx"]

then you can use 'generate_prediction(model, task, split="split_name")' to generate translation prediction
'''
#
# # Add binarized monolingual data to the original data directory, and name it with "split_name"
# # ex. ./DATA/data-bin/ted2020/\[split_name\].zh-en.\["en", "zh"\].\["bin", "idx"\]
# !cp ./DATA/data-bin/mono/train.zh-en.zh.bin ./DATA/data-bin/ted2020/mono.zh-en.zh.bin
# !cp ./DATA/data-bin/mono/train.zh-en.zh.idx ./DATA/data-bin/ted2020/mono.zh-en.zh.idx
# !cp ./DATA/data-bin/mono/train.zh-en.en.bin ./DATA/data-bin/ted2020/mono.zh-en.en.bin
# !cp ./DATA/data-bin/mono/train.zh-en.en.idx ./DATA/data-bin/ted2020/mono.zh-en.en.idx

# hint: do prediction on split='mono' to create prediction_file
# generate_prediction( ... ,split=... ,outfile=... )binpath
