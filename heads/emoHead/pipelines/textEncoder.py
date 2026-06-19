import torch
import torch.nn as nn
class TextEncoder(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(TextEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # Input: 768 -> 512
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Dropout(0.2)
            # Output: 512 -> 512
            nn.Linear(512, 512)
        )
        # Learnable silence embedding (Option B default)
        # This will be updated during backprop
        self.silence_embedding = nn.Parameter(torch.randn(512))

    def forward(self, x, is_silent_mask=None):
        # x shape: (batch_size, 768)
        # is_silent_mask: boolean tensor of shape (batch_size,)
        
        encoded_text = self.encoder(x)
        
        # Replace empty transcriptions with the learnable silence embedding
        if is_silent_mask is not None:
            # Expand silence_embedding to match batch dimensions for replacement
            batch_size = x.size(0)
            silence_batch = self.silence_embedding.unsqueeze(0).expand(batch_size, -1)
            
            # Where mask is True, use silence embedding, else use encoded text
            out = torch.where(is_silent_mask.unsqueeze(1), silence_batch, encoded_text)
            return out
            
        return encoded_text # 512-dim text_vector