import torch
import torch.nn as nn
class VoiceEncoder(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(VoiceEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # Input: 1536 -> 512
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Dropout(0.2)
            # Output: 512 -> 512
            nn.Linear(512, 512)
        )

    def forward(self, x):
        # x shape: (batch_size, 1536)
        return self.encoder(x) # 512-dim voice_vector