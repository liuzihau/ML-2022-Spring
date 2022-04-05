import json
import gc

# Pytorch
import torch

from dataset import LibriDataset
# from model.dnn_model import Classifier
from model.bidirection_lstm_model import Classifier
import train
import utils

# Configurations
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('./config/config.json', 'r', encoding="utf-8") as f:
    config = json.loads(f.read())

# # get data
train_loader, validation_loader, test_loader, train_len, validation_len = utils.load_data(config, LibriDataset)
# # fix random seed
utils.same_seeds(config["seed"])

# # create model
if config["loss"] == "sequence":
    seq = True

model = Classifier(input_dim=config["input_dim"] * config["concat_nframes"],
                   hidden_layers=config["hidden_layers"], hidden_dim=config["hidden_dim"], seq=seq).to(device)

train.trainer(train_loader, validation_loader, model, config, device, train_len, validation_len)

del train_loader, validation_loader
gc.collect()

# load model
model.load_state_dict(torch.load(config["model_path"]))
train.predict(test_loader, config, model, device)
