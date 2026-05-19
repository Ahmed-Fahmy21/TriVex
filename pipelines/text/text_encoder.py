import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=512, dropout_rate=0.2):
        super(TextEncoder, self).__init__()

        # Architecture strictly following your thesis design:
        # Linear(768 -> 512) -> ReLU -> Dropout(0.2) -> Linear(512 -> 512)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x shape expected: (batch_size, 768)
        Returns: (batch_size, 512) - text_vector
        """
        return self.encoder(x)