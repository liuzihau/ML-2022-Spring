import sys
import os
import pprint
import logging
import random
import subprocess
import time
from pathlib import Path

import torch
import numpy as np
import wandb

lib_path = os.path.abspath(os.path.join(__file__, '..', 'fairseq'))
sys.path.append(lib_path)
from fairseq import utils
from fairseq.tasks.translation import TranslationConfig, TranslationTask

from config.config import Config
from train import load_data_iterator, generate_prediction
import train
from criterion.label_smoothing_cross_entropy import LabelSmoothedCrossEntropyCriterion
from optimizer.noam_optimizer import NoamOpt

setting = Config()
config = setting.args
arch_args = setting.architecture
seed = config.seed
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
cuda_env = utils.CudaEnvironment()

utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",  # "DEBUG" "WARNING" "ERROR"
    stream=sys.stdout,
)

proj = "hw5.seq2seq"
logger = logging.getLogger(proj)
if config.use_wandb:
    wandb.init(project=proj, name=Path(config.savedir).stem, config=config)

# setup task
task_cfg = TranslationConfig(
    data=config.datadir,
    source_lang=config.source_lang,
    target_lang=config.target_lang,
    train_subset="train",
    required_seq_len_multiple=8,
    dataset_impl="mmap",
    upsample_primary=1,
)
task = TranslationTask.setup_task(task_cfg)

logger.info("loading data for epoch 1")
task.load_dataset(split="train", epoch=1, combine=True)  # combine if you have back-translation data.
task.load_dataset(split="valid", epoch=1)

sample = task.dataset("valid")[1]
pprint.pprint(sample)
pprint.pprint("Source: " + task.source_dictionary.string(sample['source'], config.post_process, ))
pprint.pprint("Target: " + task.target_dictionary.string(sample['target'], config.post_process, ))

demo_epoch_obj = load_data_iterator(task, "valid", config, epoch=1, max_tokens=20, num_workers=1, cached=False)
demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
sample = next(demo_iter)

if config.use_wandb:
    wandb.config.update(vars(arch_args))

model = train.build_model(config, arch_args, task)
logger.info(model)

# generally, 0.1 is good enough
criterion = LabelSmoothedCrossEntropyCriterion(
    smoothing=0.1,
    ignore_index=task.target_dictionary.pad(),
)
optimizer = NoamOpt(
    model_size=arch_args.encoder_embed_dim,
    factor=config.lr_factor,
    warmup=config.lr_warmup,
    optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))

model = model.to(device=device)
criterion = criterion.to(device=device)

logger.info("task: {}".format(task.__class__.__name__))
logger.info("encoder: {}".format(model.encoder.__class__.__name__))
logger.info("decoder: {}".format(model.decoder.__class__.__name__))
logger.info("criterion: {}".format(criterion.__class__.__name__))
logger.info("optimizer: {}".format(optimizer.__class__.__name__))
logger.info(
    "num. model params: {:,} (num. trained: {:,})".format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
)
logger.info(f"max tokens per batch = {config.max_tokens}, accumulate steps = {config.accum_steps}")

epoch_itr = load_data_iterator(task, "train", config, config.start_epoch, config.max_tokens, config.num_workers)
train.try_load_checkpoint(model, config, logger, optimizer, name=config.resume)
while epoch_itr.next_epoch_idx <= config.max_epoch:
    # train for one epoch
    train.train_one_epoch(epoch_itr, model, task, config, criterion, optimizer, logger, config.accum_steps)
    stats = train.validate_and_save(model, task, config, criterion, optimizer, epoch=epoch_itr.epoch, logger=logger)
    logger.info("end of epoch {}".format(epoch_itr.epoch))
    epoch_itr = load_data_iterator(task, "train", config, epoch_itr.next_epoch_idx, config.max_tokens,
                                   config.num_workers)

# averaging a few checkpoints can have a similar effect to ensemble
#####################################################
checkdir = config.savedir
subprocess.Popen(
    f"python average_checkpoints.py --inputs {checkdir} --num-epoch-checkpoints {config.average_checkpoints_number} --output {checkdir}/avg_last_5_checkpoint.pt",
    shell=True)
for i in range(10):
    print(f'averaging last 5 checkpoints , wait {10 - i} sec')
    time.sleep(1)
######################################################

# checkpoint_last.pt : latest epoch
# checkpoint_best.pt : highest validation bleu
# avg_last_5_checkpoint.pt:　the average of last 5 epochs
train.try_load_checkpoint(model, config, logger, optimizer, name="avg_last_5_checkpoint.pt")
train.validate(model, task, config, criterion, logger, log_to_wandb=False)

generate_prediction(model, task, config)
