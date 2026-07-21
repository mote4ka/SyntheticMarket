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
        last_hidden = out[:, -1, :]    
        return self.fc(last_hidden)



import pandas as pd
import numpy as np

def prepare_features_and_labels(df, horizon=5, flat_threshold=0.001):
    
    df = df.copy()
    # Absolute price difference instead of pct_change() (no division by close)
    df["return"] = df["close"].diff()
    
    # Use rolling std as the normalization scale instead of dividing by close (changed from 60 to 20)
    rolling_scale = df["close"].rolling(20, min_periods=1).std() + 1e-8
    
    df["high_low_range"] = (df["high"] - df["low"]) / rolling_scale
    df["body"] = (df["close"] - df["open"]) / rolling_scale
    
    
    df["volatility_20"] = df["return"].rolling(10, min_periods=1).std()
    
    # Also an absolute difference, not relative
    future_return = df["close"].shift(-horizon) - df["close"]
    
    def label_fn(x):
        if x > flat_threshold:
            return 2
        elif x < -flat_threshold:
            return 0
        else:
            return 1
    
    df["label"] = future_return.apply(label_fn)
    
    #print(df.isna().sum())
    #print(f"Before dropna: {len(df)}")
    df = df.dropna().reset_index(drop=True)
    #print(f"After dropna: {len(df)}")
    return df

feature_cols = ["return", "high_low_range", "body", "volatility_20"]