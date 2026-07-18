import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)      # out: (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]          # берём последний timestep
        return self.fc(last_hidden)



import pandas as pd
import numpy as np

def prepare_features_and_labels(df, horizon=5, flat_threshold=0.001):
    df = df.copy()
    
    # ВМЕСТО pct_change() (деление на close) — просто разность
    df["return"] = df["close"].diff() # абсолютная разность, не относительная
    
    # ВМЕСТО деления на close — используем rolling std как масштаб
    rolling_scale = df["close"].rolling(60).std() + 1e-8
    
    df["high_low_range"] = (df["high"] - df["low"]) / rolling_scale
    df["body"] = (df["close"] - df["open"]) / rolling_scale
    
    df["volume_norm"] = (df["volume"] - df["volume"].rolling(60).mean()) / (df["volume"].rolling(60).std() + 1e-8)
    df["volatility_20"] = df["return"].rolling(20).std()
    
    future_return = df["close"].shift(-horizon) - df["close"]   # тоже абсолютная разность, не относительная
    
    def label_fn(x):
        if x > flat_threshold:
            return 2
        elif x < -flat_threshold:
            return 0
        else:
            return 1
    
    df["label"] = future_return.apply(label_fn)
    df = df.dropna().reset_index(drop=True)
    return df

feature_cols = ["return", "high_low_range", "body", "volume_norm", "volatility_20"]