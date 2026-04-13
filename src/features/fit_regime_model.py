import numpy as np
import pandas as pd
from hmmlearn import GaussianHMM
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
import joblib

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models/clustering")

FEATURES = [
    "return_5d",
    "return_21d",
    "volatility_10d",
    "volatility_30d",
    "rsi",
    "macd",
    "sp500_vix_corr",
    "sp500_gold_corr",
    "vix_high",
    "us10y_change",
]

def train_hmm_model(data_path: Path) -> pd.DataFrame:
    """Train the HMM model and save it in /models/clustering"""
    
    df = pd.read_csv(data_path)
    
    sp500 = df[df["Ticker"] == "SP500"][["Date", "Close"] + FEATURES].copy()
    sp500 = sp500.sort_values("Date").reset_index(drop=True)
    sp500 = sp500.dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(sp500[FEATURES])

    hmm = GaussianHMM(
        n_components=5,
        covariance_type="full",
        n_iter=100,
        random_state=15
    )
    hmm.fit(X)
    joblib.dump(hmm, "./models/clustering")


def main():
    logging.basicConfig(level=logging.INFO)
    train_hmm_model(MODEL_DIR)
    logging.info(f"HMM model saved in {MODEL_DIR}")
