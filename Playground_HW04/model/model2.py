import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from model.conformer.encoder import ConformerEncoder


class Classifier(nn.Module):
    def __init__(self, config, n_spks=600):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        d_model = config['d_model']
        dropout = config['dropout']
        self.linear_num_layer = config['linear']
        self.prenet = nn.Linear(40, d_model)
        # -----
        self.conformer_encoder = ConformerEncoder(
            input_dim=d_model,
            encoder_dim=64,
            num_layers=config['num_layers'],
            num_attention_heads=config['nhead'],
            feed_forward_expansion_factor=config['feedforward'],
            conv_expansion_factor=2,
            input_dropout_p=dropout,
            feed_forward_dropout_p=dropout,
            attention_dropout_p=dropout,
            conv_dropout_p=dropout,
            conv_kernel_size=31,
            half_step_residual=True
        )
        # -----

        # Project the dimension of features from d_model into speaker nums.
        self.pred_layers = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_spks),
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

        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out,_ = self.conformer_encoder(out, out.shape[0])

        # mean pooling
        stats = out.mean(dim=1)
        # fully connected layer
        if self.linear_num_layer > 0:
            stats = self.linear_layer(stats)
        # out: (batch, n_spks)
        out = self.pred_layers(stats)
        return out
