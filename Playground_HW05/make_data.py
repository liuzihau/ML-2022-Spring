import re
import os
import random
import time
from pathlib import Path
import subprocess
import sentencepiece as spm


def strQ2B(ustring):
    """Full width -> half width"""
    # reference:https://ithelp.ithome.com.tw/articles/10233122
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # Full width space: direct conversion
                inside_code = 32
            elif 65281 <= inside_code <= 65374:  # Full width chars (except space) conversion
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s)  # remove ([text])
        s = s.replace('-', '')  # remove '-'
        s = re.sub('([.,;!?()\"])', r' \1 ', s)  # keep punctuation
    elif lang == 'zh':
        s = strQ2B(s)  # Q2B
        s = re.sub(r"\([^()]*\)", "", s)  # remove ([text])
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s)  # keep punctuation
    s = ' '.join(s.strip().split())
    return s


def len_s(s, lang):
    if lang == 'zh':
        return len(s)
    return len(s.split())


def clean_corpus(prefix_path, l1, l2, ratio=9, max_len=1000, min_len=1):
    if Path(f'{prefix_path}.clean.{l1}').exists() and Path(f'{prefix_path}.clean.{l2}').exists():
        print(f'{prefix_path}.clean.{l1} & {l2} exists. skipping clean.')
        return
    with open(f'{prefix_path}.{l1}', 'r') as l1_in_f:
        with open(f'{prefix_path}.{l2}', 'r') as l2_in_f:
            with open(f'{prefix_path}.clean.{l1}', 'w') as l1_out_f:
                with open(f'{prefix_path}.clean.{l2}', 'w') as l2_out_f:
                    for s1 in l1_in_f:
                        s1 = s1.strip()
                        s2 = l2_in_f.readline().strip()
                        s1 = clean_s(s1, l1)
                        s2 = clean_s(s2, l2)
                        s1_len = len_s(s1, l1)
                        s2_len = len_s(s2, l2)
                        if min_len > 0:  # remove short sentence
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0:  # remove long sentence
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0:  # remove by ratio of length
                            if s1_len / s2_len > ratio or s2_len / s1_len > ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)


if __name__ == "__main__":
    subprocess.Popen(f"git clone https://github.com/pytorch/fairseq.git", shell=True)
    for i in range(20):
        print(f"git clone fairseq... wait {20 - i} secs")
    subprocess.Popen(f"cd fairseq && git checkout 9a1c497", shell=True)
    for i in range(5):
        print(f"check fairseq version... wait {5 - i} secs")
    subprocess.Popen(f"pip install --upgrade ./fairseq/", shell=True)
    for i in range(20):
        print(f"upgrate fairseq... wait {20 - i} secs")

    data_dir = './DATA/rawdata'
    dataset_name = 'ted2020'
    urls = (
        "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/ted2020.tgz",
        "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/test.tgz",
    )
    file_names = (
        'ted2020.tgz',  # train & dev
        'test.tgz',  # test
    )
    prefix = Path(data_dir).absolute() / dataset_name

    prefix.mkdir(parents=True, exist_ok=True)
    for u, f in zip(urls, file_names):
        path = prefix / f
        if not path.exists():
            subprocess.Popen(f"wget {u} -O {path}", shell=True)
            for i in range(20):
                print(f'Download data... wait {20 - i} sec')
                time.sleep(1)
        if path.suffix == ".tgz":
            subprocess.Popen(f"tar -xvf {path} -C {prefix}", shell=True)
            for i in range(10):
                print(f'extract data... wait {15 - i} sec')
                time.sleep(1)
        elif path.suffix == ".zip":
            subprocess.Popen(f"unzip -o {path} -d {prefix}", shell=True)
            for i in range(10):
                print(f'extract data... wait {15 - i} sec')
                time.sleep(1)

    subprocess.Popen(f"mv {prefix / 'raw.en'} {prefix / 'train_dev.raw.en'}", shell=True)
    subprocess.Popen(f"mv {prefix / 'raw.zh'} {prefix / 'train_dev.raw.zh'}", shell=True)
    subprocess.Popen(f"mv {prefix / 'test/test.en'} {prefix / 'test.raw.en'}", shell=True)
    subprocess.Popen(f"mv {prefix / 'test/test.zh'} {prefix / 'test.raw.zh'}", shell=True)
    subprocess.Popen(f"rm -rf {prefix / 'test'}", shell=True)

    src_lang = 'en'
    tgt_lang = 'zh'

    prefix = Path(data_dir).absolute() / dataset_name
    data_prefix = f'{prefix}/train_dev.raw'
    test_prefix = f'{prefix}/test.raw'

    clean_corpus(data_prefix, src_lang, tgt_lang)
    clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)

    valid_ratio = 0.01  # 3000~4000 would suffice
    train_ratio = 1 - valid_ratio

    if (prefix / f'train.clean.{src_lang}').exists() \
            and (prefix / f'train.clean.{tgt_lang}').exists() \
            and (prefix / f'valid.clean.{src_lang}').exists() \
            and (prefix / f'valid.clean.{tgt_lang}').exists():
        print(f'train/valid splits exists. skipping split.')
    else:
        line_num = sum(1 for line in open(f'{data_prefix}.clean.{src_lang}'))
        labels = list(range(line_num))
        random.shuffle(labels)
        for lang in [src_lang, tgt_lang]:
            train_f = open(os.path.join(data_dir, dataset_name, f'train.clean.{lang}'), 'w')
            valid_f = open(os.path.join(data_dir, dataset_name, f'valid.clean.{lang}'), 'w')
            count = 0
            for line in open(f'{data_prefix}.clean.{lang}', 'r'):
                if labels[count] / line_num < train_ratio:
                    train_f.write(line)
                else:
                    valid_f.write(line)
                count += 1
            train_f.close()
            valid_f.close()

    vocab_size = 8000
    if (prefix / f'spm{vocab_size}.model').exists():
        print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train.')
    else:
        spm.SentencePieceTrainer.train(
            input=','.join([f'{prefix}/train.clean.{src_lang}',
                            f'{prefix}/valid.clean.{src_lang}',
                            f'{prefix}/train.clean.{tgt_lang}',
                            f'{prefix}/valid.clean.{tgt_lang}']),
            model_prefix=prefix / f'spm{vocab_size}',
            vocab_size=vocab_size,
            character_coverage=1,
            model_type='unigram',  # 'bpe' works as well
            input_sentence_size=1e6,
            shuffle_input_sentence=True,
            normalization_rule_name='nmt_nfkc_cf',
        )

    spm_model = spm.SentencePieceProcessor(model_file=str(prefix / f'spm{vocab_size}.model'))
    in_tag = {
        'train': 'train.clean',
        'valid': 'valid.clean',
        'test': 'test.raw.clean',
    }
    for split in ['train', 'valid', 'test']:
        for lang in [src_lang, tgt_lang]:
            out_path = prefix / f'{split}.{lang}'
            if out_path.exists():
                print(f"{out_path} exists. skipping spm_encode.")
            else:
                with open(prefix / f'{split}.{lang}', 'w') as out_f:
                    with open(prefix / f'{in_tag[split]}.{lang}', 'r') as in_f:
                        for line in in_f:
                            line = line.strip()
                            tok = spm_model.encode(line, out_type=str)
                            print(' '.join(tok), file=out_f)

    binpath = Path('../DATA/data-bin', dataset_name)
    if binpath.exists():
        print(binpath, "exists, will not overwrite!")
    else:
        subprocess.Popen(
            f"python -m fairseq_cli.preprocess --source-lang {src_lang} --target-lang {tgt_lang} --trainpref {prefix / 'train'} --validpref {prefix / 'valid'} --testpref {prefix / 'test'} --destdir {binpath} --joined-dictionary --workers 2",
            cwd="fairseq", shell=True)
    subprocess.Popen(f"python setup.py build_ext --inplace", cwd="fairseq", shell=True)
