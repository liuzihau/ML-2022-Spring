import os
import random
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_feat(path):
    feat = torch.load(path)
    return feat


def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)


def concat_feat(x, concat_n):
    assert concat_n % 2 == 1  # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2)  # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid + 1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


def preprocess_data(split, feat_dir, phone_path, concat_nframes, config, train_ratio=0.8, train_val_seed=1337):
    class_num = 41  # NOTE: pre-computed, should not need change
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
        phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

        for line in phone_file:
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(
        len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
        if config["loss"] == "CTC":
            y = torch.empty(max_len, concat_nframes, dtype=torch.long)
        elif config["loss"] == "sequence":
            y = torch.empty(max_len, concat_nframes, dtype=torch.long)
        else:
            y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
            if config["loss"] == "CTC":
                label = torch.LongTensor(label_dict[fname])
                label = label.view(-1, 1)
                label = concat_feat(label, concat_nframes)
            elif config["loss"] == "sequence":
                label = torch.LongTensor(label_dict[fname])
                label = label.view(-1, 1)
                label = concat_feat(label, concat_nframes)
            else:
                label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
            if config["loss"] == "CTC":
                y[idx: idx + cur_len, :] = label
            elif config["loss"] == "sequence":
                y[idx: idx + cur_len, :] = label
            else:
                y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode != 'test':
        y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
        print(y.shape)
        return X, y
    else:
        return X


def load_data(config, dataset):
    # preprocess data
    train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone',
                                       concat_nframes=config["concat_nframes"], train_ratio=config["train_ratio"],
                                       config=config)
    val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone',
                                   concat_nframes=config["concat_nframes"], train_ratio=config["train_ratio"],
                                   config=config)
    test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone',
                             concat_nframes=config["concat_nframes"], config=config)

    # get dataset
    train_set = dataset(train_X, train_y)
    validation_set = dataset(val_X, val_y)
    test_set = dataset(test_X, None)
    # remove raw feature to save memory
    del train_X, train_y, val_X, val_y
    gc.collect()

    # get dataloader
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    return train_loader, validation_loader, test_loader, len(train_set), len(validation_set)


def save_prediction(prediction):
    datetime_today_obj = datetime.datetime.today()
    today_string = datetime_today_obj.strftime('%Y_%m_%d')
    submission = pd.DataFrame(columns=['id', 'tested_positive'])
    submission['id'] = np.arange(prediction.shape[0])
    submission['tested_positive'] = prediction
    submission.to_csv(f'prediction_{today_string}.csv', index=False)
