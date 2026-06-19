import torch
import torch.nn as nn

class BodyEncoder(nn.Module):
    def __init__(self):
        super(BodyEncoder, self).__init__()
        # Input: 132 -> 256
        self.fc1 = nn.Linear(132, 256)
        # LSTM: 1 layer, 256 hidden
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        # Output mapping: 256 -> 256
        self.fc2 = nn.Linear(256, 256)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, 132)
        x = self.fc1(x)
        lstm_out, (hn, cn) = self.lstm(x)
        # Taking the last hidden state of the sequence
        last_hidden = hn[-1] # shape: (batch_size, 256)
        out = self.fc2(last_hidden)
        return out # 256-dim body_vector