import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
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

def detect_regimes(features_path: Path) -> pd.DataFrame:
    """Apply the HMM model to the data"""
    try:
        df = pd.read_csv(PROCESSED_DIR / "market_features.csv")

        hmm = joblib.load(MODEL_DIR / "hmm.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")

        sp500 = df[df["Ticker"] == "SP500"][["Date", "Close"] + FEATURES].copy()
        sp500 = sp500.sort_values("Date").reset_index(drop=True)
        sp500 = sp500.dropna()

        sp500["regime"] = hmm.predict(scaler.transform(sp500[FEATURES]))

        df = df.merge(sp500[["Date", "regime"]], on="Date", how="left")
        
        return df
    
    except Exception as e:
        logging.error(f"Error during clustering: {e}")
        return None
    


def main():
    logging.basicConfig(level=logging.INFO)
    df = detect_regimes(PROCESSED_DIR / "market_features.csv")
    df.to_csv(PROCESSED_DIR / "market_regimes.csv", index=False)
    logging.info(f"Regimes saved")


if __name__ == "__main__":
    main()