import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader


def same_seed(seed):
    """
    Fixes random number generator seeds for reproducibility.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    """
    Split provided training data into training set and validation set
    """
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def select_feat(train_data, valid_data, test_data, select_all=True):
    """Selects useful features to perform regression"""
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        # Nick -16+ 1.1538 0.94
        # feat_idx = [-16, -15, -14, -13, -12, -9, -4, -3, -2]
        # Nick -16+ 5
        feat_idx = [-16, -15, -14, -13, -12]

        # feat_idx = [0, 1, 2, 3, 4]  # TODO: Select suitable feature columns.

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


def load_data(config, train_data, test_data, dataset):
    # Dataloader
    # Set seed for reproducibility
    same_seed(config['seed'])

    # train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
    # test_data size: 1078 x 117 (without last day's positive rate)

    train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

    # Print out the data size.
    print(f"""train_data size: {train_data.shape} 
    valid_data size: {valid_data.shape} 
    test_data size: {test_data.shape}""")

    # Select features
    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])
    # Print out the number of features.
    print(f'number of features: {x_train.shape[1]}')

    train_dataset, valid_dataset, test_dataset = (dataset(x_train, y_train),
                                                  dataset(x_valid, y_valid),
                                                  dataset(x_test))

    # Pytorch data loader loads pytorch dataset into batches.
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    input_dims = x_train.shape[1:]
    return train_loader, valid_loader, test_loader, input_dims


def save_prediction(prediction):
    datetime_today_obj = datetime.datetime.today()
    today_string = datetime_today_obj.strftime('%Y_%m_%d')
    submission = pd.DataFrame(columns=['id', 'tested_positive'])
    submission['id'] = np.arange(prediction.shape[0])
    submission['tested_positive'] = prediction
    submission.to_csv(f'prediction_{today_string}.csv', index=False)
