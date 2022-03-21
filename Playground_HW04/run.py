import json

import utils
import train

"""arguments"""
with open('./config/config.json', 'r', encoding="utf-8") as config_json:
    config = json.loads(config_json.read())


utils.set_seed(config['seed'])
train_loader, valid_loader, test_loader, speaker_num = utils.get_dataloader(config)
train.train_the_model(config, train_loader, valid_loader, speaker_num)
train.predict_and_save_result(config, test_loader)
