import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding used for time-series."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # even channels
        pe[:, 1::2] = torch.cos(position * div_term)   # odd channels

        self.pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


class TimeSeriesTransformer(nn.Module):
    """
    A minimal Time-Series Transformer for forecasting.
    Encoder-only architecture with regression output.
    """
    def __init__(
        self,
        input_dim,
        d_model=64,
        n_heads=4,
        num_layers=3,
        dropout=0.1,
        forecast_horizon=5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        # project input features → transformer embedding dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # simple regression head
        self.fc_out = nn.Linear(d_model, forecast_horizon)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, forecast_horizon)
        """
        x = self.input_projection(x)  # → (batch, seq_len, d_model)
        x = self.positional_encoding(x)

        encoded = self.encoder(x)  # (batch, seq_len, d_model)

        # use the last token representation for forecasting
        last_token = encoded[:, -1, :]  # (batch, d_model)

        output = self.fc_out(last_token)  # (batch, forecast_horizon)
        return output

