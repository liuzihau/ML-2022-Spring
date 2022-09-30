import json

import utils
import train

"""arguments"""
with open('./config/config.json', 'r', encoding="utf-8") as config_json:
    config = json.loads(config_json.read())

utils.set_seed(config['seed'])

train_loader, valid_loader, test_loader, test_loader2, speaker_num = utils.get_dataloader(config)

tgt_encoder = train.train_adaptation_model(config, train_loader, test_loader2, speaker_num)
train.predict_and_save_result_with_adda(config, test_loader, tgt_encoder)
