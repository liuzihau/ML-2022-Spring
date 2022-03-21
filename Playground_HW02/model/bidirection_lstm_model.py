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


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()
        self.input_dimension = int(39)
        self.sequence_length = int(input_dim/39)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_size = hidden_dim
        self.num_layers = hidden_layers
        self.lstm = nn.LSTM(self.input_dimension, hidden_dim, hidden_layers, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        hidden = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out, hidden = self.rnn(x, hidden)
        out, hidden = self.lstm(x, (hidden, c))
        # (batch_size, seq_length, hidden_size)-->(batch_size,hidden_size)
        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)

        return out
