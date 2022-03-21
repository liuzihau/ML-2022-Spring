import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=True):
        super(BasicBlock, self).__init__()
        self.drop = dropout
        self.hidden_size = output_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers=1, batch_first=True,
                            bidirectional=True, dropout=0)
        self.layer_norm = nn.LayerNorm(2 * output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        hidden = torch.zeros(1 * 2, x.size(0), self.hidden_size).to(self.device)
        c = torch.zeros(1 * 2, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.lstm(x, (hidden, c))
        x = self.layer_norm(x)
        if self.drop:
            x = self.dropout(x)

        return x


class Classifier5(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, ctc=False, seq=False):
        super(Classifier5, self).__init__()
        self.ctc = ctc
        self.seq = seq
        self.input_dimension = int(39)
        self.sequence_length = int(input_dim / 39)
        self.hidden_size = hidden_dim
        self.num_layers = hidden_layers
        self.basic_block = nn.Sequential(
            BasicBlock(self.input_dimension, hidden_dim),
            *[BasicBlock(2 * hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            BasicBlock(2 * hidden_dim, hidden_dim,
                       dropout=False)
        )
        if self.ctc:
            self.fc = nn.Linear(hidden_dim * 2, output_dim + 1)
        else:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out, hidden = self.rnn(x, hidden)
        out = self.basic_block(x)
        # (batch_size, seq_length, hidden_size)-->(batch_size,hidden_size)
        if self.ctc or self.seq:
            out = self.fc(out.contiguous())
        else:
            out = self.fc(out[:, -1, :])

        return out
