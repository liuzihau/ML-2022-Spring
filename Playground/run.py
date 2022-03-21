# Reading/Writing Data
import pandas as pd
# Pytorch
import torch
# Nick分裝
import dataset
from dnn_model import DNNModel
import train
import utils

# Configurations
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 2022,  # Your seed number, you can pick your lucky number. :)
    # Nick : True --> False
    'select_all': False,  # Whether to use all features.
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 100,  # Number of epochs.
    # Nick : 256 --> 16
    'batch_size': 16,
    # Nick 1e-6 1.15 -->1e-7 1.20
    'learning_rate': 1e-6,
    'early_stop': 400,  # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
# get data
train_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values
# on load for pytorch
train_loader, valid_loader, test_loader, input_dim = utils.load_data(config, train_data, test_data, dataset.COVID19Dataset)
# Start training!
model = DNNModel(input_dim=input_dim).to(device)  # put your model and data on the same computation device.
train.trainer(train_loader, valid_loader, model, config, device)
model.load_state_dict(torch.load(config['save_path']))
prediction = train.predict(test_loader, model, device)
utils.save_prediction(prediction)


# import xgboost
# from xgboost import XGBClassifier
#
# print(x_train.shape)  ## (445, 17)
# print(x_valid.shape)  ## (446, 17)
# print(y_train.shape)  ## (445,)
# print(y_valid.shape)  ## (446,)
#
# # Set our parameters for xgboost
# params = {}
# # 請填入以下參數:
# # 目標函數: 二元分類
# # 評價函數: logloss
# # 學習速度: 0.04
# # 最大深度: 5
# #=============your works starts===============#
# params['objective'] = 'reg:squarederror'
# params['eval_metric'] = 'rmse'
# params['eta'] = 0.0005
# params['max_depth'] = 2
# #==============your works ends================#
#
# d_train = xgboost.DMatrix(x_train, label=y_train)
# d_valid = xgboost.DMatrix(x_valid, label=y_valid)
#
# watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#
# bst = xgboost.train(params, d_train, 120000, watchlist, early_stopping_rounds=100, verbose_eval=1)
# y_pred = bst.predict(xgboost.DMatrix(x_valid))
# print(y_valid.shape)
# print("Accuracy: ", str((sum((y_valid - y_pred) ** 2)/(y_valid.shape[0])) ** (1/2)))
# y_pred2 = bst.predict(xgboost.DMatrix(x_test))
# print(y_pred2.shape)
# submission = pd.DataFrame(columns=['id','tested_positive'])
# submission['id'] = pd.read_csv('covid.test.csv')['id']
# submission['tested_positive'] = y_pred2
# submission.to_csv('prediction.csv',index=False)
