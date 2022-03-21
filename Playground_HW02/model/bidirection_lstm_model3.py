import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier3(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier3, self).__init__()
        self.input_dimension = int(39)
        self.sequence_length = int(input_dim / 39)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_size = hidden_dim
        self.num_layers = hidden_layers
        self.lstm = nn.LSTM(self.input_dimension, hidden_dim, hidden_layers, batch_first=True,
                            bidirectional=True, dropout=0.15)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        hidden = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out, hidden = self.rnn(x, hidden)
        out, hidden = self.lstm(x, (hidden, c))
        # (batch_size, seq_length, hidden_size)-->(batch_size,hidden_size)
        out = self.fc(out[:, -1, :])

        return out
