import torch.nn as nn
import torch


class BidirectionLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BidirectionLSTMModel, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='relu', batch_first=True)
        self.fc = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
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
