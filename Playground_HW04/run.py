import json

import utils
import train

"""arguments"""
with open('./config/config.json', 'r', encoding="utf-8") as config_json:
    config = json.loads(config_json.read())

utils.set_seed(config['seed'])
train_loader = {}
valid_loader = {}
for i in range(5):
    train_loader[f'{i}'], valid_loader[f'{i}'], test_loader, test_loader2, speaker_num = utils.get_dataloader(config, i)
    train.train_the_model(config, train_loader[f'{i}'], valid_loader[f'{i}'], speaker_num, i)
    if config['domain_adaptation']:
        train.train_adaptation_model(config, train_loader[f'{i}'], test_loader2, speaker_num)
        train.test_and_save_result()
    train.predict_and_save_result(config, test_loader, i)
