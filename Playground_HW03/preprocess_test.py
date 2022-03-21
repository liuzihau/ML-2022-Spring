# Import necessary packages.
import numpy as np
import json
import cv2
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torchvision
from dataset import FoodDataset
from model.sample_model import Classifier
import utils
from preprocess import transform
from optim.opt_lookahead import Lookahead
from torch_optimizer import SGDW
import train

with open('./config/config.json', 'r', encoding="utf-8") as config_json:
    config = json.loads(config_json.read())
# "cuda" only when GPUs are available.
config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config["seed"])

tfm = transform.transform(config)

train_set, valid_set, test_set, loader = utils.load_data([FoodDataset], config, tfm)

train_loader = loader['train']

for batch in tqdm(train_loader):
    input_images, input_labels = batch
    # output_images,output_labels = transform.transform_timm(input_images,input_labels,config)
    output_image = torchvision.utils.make_grid(input_images)
    cv2.imshow('test', output_image.permute(1, 2, 0).to('cpu').detach().numpy()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break
