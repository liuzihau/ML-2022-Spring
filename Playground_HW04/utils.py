import math
import numpy as np
import random

import torch
from dataset import myDataset, InferenceDataset, InferenceDataset2
from torch.utils.data import DataLoader, random_split, Subset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collate_batch(batch):
    # Process features within a batch.
    """Collate a batch of data."""
    mel, speaker = zip(*batch)
    # train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)  # pad log 10^(-20) which is very small value.
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()


def collate_batch_for_adaptation(batch):
    # Process features within a batch.
    """Collate a batch of data."""
    feat_paths, mel = zip(*batch)
    # train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)  # pad log 10^(-20) which is very small value.
    # mel: (batch size, length, 40)
    return mel, feat_paths


def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)

    return feat_paths, torch.stack(mels)


def get_dataloader(config, i=None):
    """Generate dataloader"""
    data_dir = config['data_dir']
    batch_size = config['batch_size']
    n_workers = config['n_workers']
    dataset = myDataset(data_dir, config)
    testset = InferenceDataset(config)
    testset2 = InferenceDataset2(config)

    speaker_num = dataset.get_speaker_number()
    # Split dataset into training dataset and validation dataset
    if i!=None:
        trainset = Subset(dataset, [k for k in range(len(dataset)) if k % 5 != i])
        validset = Subset(dataset, [k for k in range(len(dataset)) if k % 5 == i])
    else:
        trainset = Subset(dataset, [k for k in range(len(dataset)) if k % 5 != 4])
        validset = Subset(dataset, [k for k in range(len(dataset)) if k % 5 == 4])

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=n_workers,
        collate_fn=inference_collate_batch,
    )
    test_loader2 = DataLoader(
        testset2,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch_for_adaptation,
    )

    return train_loader, valid_loader, test_loader, test_loader2,speaker_num


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
        The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
        The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
        The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
        The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
        following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
        The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
