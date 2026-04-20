import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
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

def train_hmm_model(data_path: Path) -> None:
    """Train the HMM model and save it in /models/clustering"""
    
    try:
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

        sp500["regime"] = hmm.predict(X)
        regime_vol = sp500.groupby("regime")["volatility_30d"].mean()
        sorted_regimes = regime_vol.sort_values().index
        mapping = {int(old):int(new) for new, old in enumerate(sorted_regimes)}

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(hmm, MODEL_DIR / "hmm.pkl")
        joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
        joblib.dump(scaler, MODEL_DIR / "regime_mapping.pkl")

    except Exception as e:
        logging.error(f"Error training the model: {e}")
        return None


def main():
    logging.basicConfig(level=logging.INFO)
    train_hmm_model(PROCESSED_DIR / "market_features.csv")
    logging.info(f"HMM model saved in {MODEL_DIR}")


if __name__ == "__main__":
    main()