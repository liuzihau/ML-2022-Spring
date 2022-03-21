import json
# Pytorch
import torch

from dataset import LibriDataset
# from model.dnn_model import Classifier
# from model.bidirection_lstm_model import Classifier
from model.bidirection_lstm_model5 import Classifier5

import train
import utils
import gc

# Configurations
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('./config/config.json', 'r', encoding="utf-8") as f:
    config = json.loads(f.read())

# # get data
train_loader, validation_loader, test_loader, train_len, validation_len = utils.load_data(config, LibriDataset)
# # fix random seed
utils.same_seeds(config["seed"])
#
# # create model
if config["loss"] == "CTC":
    ctc = True
elif config["loss"] == "sequence":
    ctc = False
    seq = True
else:
    ctc = False

model = Classifier5(input_dim=config["input_dim"] * config["concat_nframes"],
                    hidden_layers=config["hidden_layers"], hidden_dim=config["hidden_dim"], ctc=ctc, seq=seq).to(device)

train.trainer(train_loader, validation_loader, model, config, device, train_len, validation_len)

del train_loader, validation_loader
gc.collect()

# load model
model.load_state_dict(torch.load(config["model_path"]))
train.predict(test_loader, config, model, device)
