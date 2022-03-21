import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, config, n_spks=600):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        d_model = config['d_model']
        dropout = config['dropout']
        self.linear_num_layer = config['linear']
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=config['feedforward'],
            nhead=config['nhead'],
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config['num_layers'])

        # Project the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(

                nn.Linear(d_model, d_model),
                nn.ReLU(),
        nn.Linear(d_model, n_spks)
        )


    def forward(self, mels):
        """
        args:
            mels: (batch size, length, 40)
        return:
            out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)
        # fully connected layer
        if self.linear_num_layer > 0:
            stats = self.linear_layer(stats)
        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out
