import torch
import torch.nn as nn


class BodyEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 132,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        batch_size, T, _ = x.shape

        x = self.input_projection(x)

        lstm_out, (h_n, c_n) = self.lstm(x)

        last_hidden = h_n[0]

        body_vector = self.output_projection(last_hidden)

        if squeeze:
            body_vector = body_vector.squeeze(0)

        return body_vector

    def get_trainable_params(self) -> int:

        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_body_encoder(device: str = "cpu") -> BodyEncoder:

    encoder = BodyEncoder(
        input_dim=132,
        hidden_dim=256,
        output_dim=256,
        dropout=0.2,
    )
    encoder = encoder.to(device)
    print(f"✅ Body Encoder Ready| Trainable params: {encoder.get_trainable_params():,}")
    return encoder

if __name__ == "__main__":
    print("=" * 50)
    print("Body Encoder Test")
    print("=" * 50)

    encoder = build_body_encoder(device="cpu")

    dummy_input = torch.randn(4, 150, 132)
    print(f"\nInput shape:  {dummy_input.shape}")

    output = encoder(dummy_input)
    print(f"Output shape: {output.shape}")

    single = torch.randn(150, 132)
    out_single = encoder(single)
    print(f"Single input → output shape: {out_single.shape}")

    print("\n✅ Body Encoder Works!")
    print(f"   Input:  (batch, T, 132)")
    print(f"   Output: (batch, 256)")
