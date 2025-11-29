import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.models.transformer import TimeSeriesTransformer
from src.data.preprocess import (
    load_health_data,
    interpolate_missing,
    create_sliding_windows,
    scale_data,
)


def train(config_path):

    # -------------------------
    # Load config
    # -------------------------
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_path = cfg["data"]["path"]
    input_len = cfg["data"]["input_len"]
    horizon = cfg["model"]["forecast_horizon"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["learning_rate"]

    # -------------------------
    # Load data
    # -------------------------
    df = load_health_data(data_path)
    df = interpolate_missing(df, column="value")

    values = df["value"].values.astype(float)

    # -------------------------
    # Create windows
    # -------------------------
    X, y = create_sliding_windows(values, input_len=input_len, forecast_horizon=horizon)

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train_scaled, X_test_scaled, scaler = scale_data(
        X_train, X_test
    )

    # -------------------------
    # Make tensors + dataloaders
    # -------------------------
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # -------------------------
    # Initialize model
    # -------------------------
    model = TimeSeriesTransformer(
        input_dim=1,
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        forecast_horizon=horizon
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            preds = model(batch_x)
            loss = criterion(preds, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    train(args.config)

