# import numpy as np
# import datetime
# import gc
#
# datetime_today_obj = datetime.datetime.today()
# today_string = datetime_today_obj.strftime('%Y_%m_%d_(%H:%M:%S)')
# print(np.arange(50))
# print(today_string)
#
# a = np.array([[[1, 2, 3, 4], [2, 2, 3, 4], [3, 2, 4, 5]], [[1, 2, 3, 4], [2, 2, 3, 4], [3, 2, 4, 5]],
#               [[1, 2, 3, 4], [2, 2, 3, 4], [3, 2, 4, 5]]])
# print(a.shape[2:])
#
# import torch
#
# # loss = MaskedSoftmaxCELoss()
# # loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
# #      torch.tensor([4, 2, 0]))
# print(torch.ones(3, 4, 10))
# print(torch.ones(3, 4))
#
# # 建立DNN神經網路
# import torch
# from torch import nn
# import torch.nn.functional as F
#
# # 無論如何一定要先確定是cuda還是cpu
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# # 建立DNN神經網路
# import torch
# from torch import nn
# import torch.nn.functional as F
#
# # 無論如何一定要先確定是cuda還是cpu
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
#
# class DNNSequential(nn.Module):
#     def __init__(self):
#         super(DNNSequential, self).__init__()
#         self.sq = nn.Sequential(nn.Linear(784, 500),
#                                 nn.BatchNorm1d(500),
#                                 nn.ReLU(),
#                                 nn.Dropout(0.25),
#                                 nn.Linear(500, 250),
#                                 nn.BatchNorm1d(250),
#                                 nn.ReLU(),
#                                 nn.Dropout(0.25),
#                                 nn.Linear(250, 125),
#                                 nn.BatchNorm1d(125),
#                                 nn.ReLU(),
#                                 nn.Dropout(0.25),
#                                 nn.Linear(125, 10),
#                                 nn.Softmax())
#
#     def forward(self, x):
#         return self.sq(x)
#
#
# model = DNNSequential().to(device)
import torch

a = torch.tensor([[1,2,3]])
b = torch.tensor([[2,2,3]])
c = torch.cat((a,b),0)
print(a.shape)
print(c.shape)
print(c[0])
d = torch.tensor([[3,0,0]])
c = torch.cat((c,d),0)
print(c)