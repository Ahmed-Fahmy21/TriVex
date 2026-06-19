import torch
import torch.nn as nn 
class FaceEncoder(nn.Module):
    def __init__(self):
        super(FaceEncoder, self).__init__()
        # Input: 1916 -> 384
        self.fc1 = nn.Linear(1916, 384)
        # LSTM: 1 layer, 384 hidden
        self.lstm = nn.LSTM(input_size=384, hidden_size=384, num_layers=1, batch_first=True)
        # Output mapping: 384 -> 384
        self.fc2 = nn.Linear(384, 384)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, 1916)
        x = self.fc1(x)
        lstm_out, (hn, cn) = self.lstm(x)
        # Taking the last hidden state
        last_hidden = hn[-1] # shape: (batch_size, 384)
        out = self.fc2(last_hidden)
        return out # 384-dim face_vector