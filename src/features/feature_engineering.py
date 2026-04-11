import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import requests

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def fetch_fred_series(series_id: str, start: str = "2000-01-01") -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, parse_dates=["DATE"], index_col="DATE")
    df = df[df.index >= start]
    df.columns = [series_id]
    return df[series_id].replace(".", np.nan).astype(float)

def add_basic_features(file_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(file_path, parse_dates=["Date"])
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        # SEASONALITY FEATURES
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["month"] = df["Date"].dt.month
        df["quarter"] = df["Date"].dt.quarter

        # RETURNS
        df["return"] = df.groupby("Ticker")["Close"].pct_change()
        df["log_return"] = df.groupby("Ticker")["Close"].transform(lambda x: np.log(x / x.shift(1)))
        df["return_5d"] = df.groupby("Ticker")["Close"].pct_change(5)
        df["return_21d"] = df.groupby("Ticker")["Close"].pct_change(21)
        df["cum_return"] = df.groupby("Ticker")["return"].transform(lambda x: (1 + x).cumprod())

        # VOLATILITY
        df["volatility_10d"] = (
            df.groupby("Ticker")["return"]
            .rolling(10)
            .std()
            .reset_index(level=0, drop=True)
        )

        df["volatility_30d"] = (
            df.groupby("Ticker")["return"]
            .rolling(30)
            .std()
            .reset_index(level=0, drop=True)
        )

        # SMA
        df["sma_20"] = (
            df.groupby("Ticker")["Close"]
            .rolling(20)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["sma_50"] = (
            df.groupby("Ticker")["Close"]
            .rolling(50)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["sma_200"] = (
            df.groupby("Ticker")["Close"]
            .rolling(200)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # EMA
        df["ema_20"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=20).mean())
        df["ema_50"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=50).mean())

        # RSI
        delta = df.groupby("Ticker")["Close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.groupby(df["Ticker"]).transform(lambda x: x.rolling(14).mean())
        avg_loss = loss.groupby(df["Ticker"]).transform(lambda x: x.rolling(14).mean())

        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=12).mean())
        ema26 = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=26).mean())

        df["macd"] = ema12 - ema26
        df["macd_signal"] = df.groupby("Ticker")["macd"].transform(lambda x: x.ewm(span=9).mean())

        # BOLLINGER
        rolling_mean = (
            df.groupby("Ticker")["Close"]
            .rolling(20)
            .mean()
            .reset_index(level=0, drop=True)
        )

        rolling_std = (
            df.groupby("Ticker")["Close"]
            .rolling(20)
            .std()
            .reset_index(level=0, drop=True)
        )

        df["bollinger_upper"] = rolling_mean + 2 * rolling_std
        df["bollinger_lower"] = rolling_mean - 2 * rolling_std

        return df
    
    except Exception as e:
        logging.error(f"Error in basic features: {e}")
        return df


def add_cross_market_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Pivot to wide format: one column per ticker
        pivot = df.pivot_table(index="Date", columns="Ticker", values="Close")

        cross = pd.DataFrame(index=pivot.index)

        # Rolling correlations (60 days)
        cross["sp500_vix_corr"]  = pivot["SP500"].rolling(60).corr(pivot["VIX"])
        cross["sp500_gold_corr"] = pivot["SP500"].rolling(60).corr(pivot["GOLD"])
        cross["sp500_oil_corr"]  = pivot["SP500"].rolling(60).corr(pivot["OIL"])

        # Spreads / ratios
        cross["gold_dollar_spread"] = pivot["GOLD"] - pivot["DOLLAR"]
        cross["gold_oil_ratio"]     = pivot["GOLD"] / pivot["OIL"]

        # Flags (only VIX for now)
        cross["vix_high"] = (pivot["VIX"] > 30).astype(int)

        # US10Y daily change
        cross["us10y_change"] = pivot["US10Y"].pct_change()

        # Merge back into long format on Date
        cross = cross.reset_index()
        df = df.merge(cross, on="Date", how="left")

        logging.info(f"Cross-market features added: {list(cross.columns)}")
        return df

    except Exception as e:
        logging.error(f"Error in cross-market features: {e}")
        return df

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        yield_curve = fetch_fred_series("T10Y2Y")   # 10Y - 2Y spread
        
        macro = pd.DataFrame({
            "yield_curve_spread": yield_curve,
        })
        macro.index.name = "Date"
        macro = macro.reset_index()
        
        df = df.merge(macro, on="Date", how="left")
        df["yield_curve_spread"] = df["yield_curve_spread"].ffill()
        
        logging.info("Macro features added: yield_curve_spread")
        return df
    except Exception as e:
        logging.error(f"Error fetching FRED data: {e}")
        return None

def main():

    logging.basicConfig(level=logging.INFO)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DIR / "market_data.csv"

    if not file_path.exists():
        logging.error("market_data.csv not found")
        return

    logging.info("Creating features")

    df = add_basic_features(file_path=file_path)
    df = add_cross_market_features(df=df)
    df = add_macro_features(df=df)

    if df is None:
        logging.error("Feature creation failed")
        return

    output_path = PROCESSED_DIR / "market_features.csv"

    df.to_csv(output_path, index=False)

    logging.info(f"Features saved in {output_path}")


if __name__ == "__main__":
    main()