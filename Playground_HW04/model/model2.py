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
        self.prenet = nn.Linear(40, d_model)
        # -----
        self.conformer_encoder = ConformerEncoder(
            input_dim=d_model,
            encoder_dim=config['encoder_dim'],
            num_layers=config['num_layers'],
            num_attention_heads=config['nhead'],
            feed_forward_expansion_factor=config['feedforward'],
            conv_expansion_factor=config['conv_expansion_factor'],
            input_dropout_p=config['dropout_input'],
            feed_forward_dropout_p=config['dropout_feedforward'],
            attention_dropout_p=config['dropout_attn'],
            conv_dropout_p=config['dropout_conv'],
            conv_kernel_size=config['conv_kernel_size'],
            half_step_residual=True
        )
        # -----
        self.pooling = None
        self.pooling_flag = False
        if config['pooling'] == 'self_attention_pooling':
            self.pooling_flag = True
            from model.pooling.self_attention_pooling import SelfAttentionPooling
            self.pooling = SelfAttentionPooling(config['encoder_dim'])

        # Project the dimension of features from d_model into speaker nums.
        self.pred_layers = nn.Sequential(
            nn.Linear(config['encoder_dim'], n_spks),
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
        out, _ = self.conformer_encoder(out, out.shape[0])
        if self.pooling_flag:
            stats = self.pooling(out)
        else:
            # mean pooling
            stats = out.mean(dim=1)

        # out: (batch, n_spks)
        out = self.pred_layers(stats)
        return out
