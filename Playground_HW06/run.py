from utils import same_seeds
from train.trainer import TrainerGAN
from config.config import Config

train = False
inference = True

same_seeds(2022)
load_setting = Config()
config = load_setting.args
trainer = TrainerGAN(config)

if train:
    trainer.train()
if inference:
    trainer.inference(f'{config["workspace_dir"]}/checkpoints/G_500.pth')
