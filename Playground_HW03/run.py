import numpy as np
import json

import torch
import torch.nn as nn
from torch_optimizer import SGDW

from preprocess import transform2 as transform
from dataset import FoodDataset, FoodDatasetMixUp, FoodDatasetEnsemble
import train
import utils
from optim.opt_lookahead import Lookahead
from optim.schedule import get_cosine_schedule_with_warmup

with open('./config/config.json', 'r', encoding="utf-8") as config_json:
    config = json.loads(config_json.read())
if config['model_name'] == "convnext":
    from model.sample_model2 import Classifier

elif config['model_name'] == "convnextv2":
    from model.sample_model3 import Classifier
elif config['model_name'] == "sample":
    from model.sample_model import Classifier

config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config["seed"])

tfm = transform.transform(config)

concat_dataset = [FoodDataset] # , FoodDatasetMixUp]
# train_set, valid_set, test_set, loader = utils.load_data(concat_dataset, FoodDatasetEnsemble, config, tfm)
test_set, loader = utils.load_data_from_folder(concat_dataset, FoodDatasetEnsemble, config, tfm)


for i in range(config["cross_valid"]):

    model = Classifier(config)
    model = model.to(config["device"])
    if not i:
        model.load_state_dict(torch.load(f"{config['_exp_name']}_best{i}.ckpt"))

    # For the classification task, we use cross-entropy as the measurement of performance.
    if config["criterion"] == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    elif config["criterion"] == "CrossEntropy_MixUp":
        pass


    # Initialize optimizer, you may fine-tune some hyper_parameters such as learning rate on your own.
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"],
                                    weight_decay=config["weight_decay"])
    elif config["optimizer"] == "SGDW":
        optimizer = SGDW(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"],
                         weight_decay=config["weight_decay"])

    if config["lookahead"] == 1:
        optimizer = Lookahead(optimizer)

    config[f'best_acc_{i}'] = train.train_the_model(model, loader, config, criterion, optimizer, split=i)
    model.load_state_dict(torch.load(f"{config['_exp_name']}_best{i}.ckpt"))
    train.do_prediction(model, config, loader["test"][0], test_set)
