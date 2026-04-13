import yfinance as yf
import pandas as pd
import os
import logging
from pathlib import Path

TICKERS = {
    "SP500":     "^GSPC",
    "NASDAQ":    "^IXIC",
    "DOW":       "^DJI",
    "CAC40":     "^FCHI",
    "NIKKEI":    "^N225",
    "EUROSTOXX": "^STOXX50E",
    "VIX":       "^VIX",
    "GOLD":      "GC=F",
    "OIL":       "CL=F",
    "US10Y":     "^TNX",
    "DOLLAR":    "DX-Y.NYB",
}

RAW_DIR = Path("data/raw")
START_DATE = "2000-01-01"

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(), # Send logs into the terminal
            logging.FileHandler("logs/fetch_market_data.log"), # Send logs into the log file
        ],
    )

def download_ticker(name: str, symbol: str) -> pd.DataFrame | None:
    """Downloading data from START_DATE"""
    try:
        logging.info(f"[{name}] Complete download from {START_DATE}")
        df = yf.download(symbol, start=START_DATE, progress=False)
        if df.empty:
            logging.warning(f"[{name}] No data returned")
            return None
        df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["Ticker"] = name
        logging.info(f"[{name}] {len(df)} lines downloaded")
        return df
    except Exception as e:
        logging.error(f"[{name}] Error downloading: {e}")
        return None

def update_ticker(name: str, symbol: str, file_path: Path) -> pd.DataFrame | None:
    """Incremental update"""
    try:
        old_df = pd.read_csv(file_path, parse_dates=["Date"])

        last_date = old_df["Date"].max()
        start = last_date + pd.Timedelta(days=1)
        logging.info(f"[{name}] Update from {start.date()}")

        new_df = yf.download(symbol, start=start, progress=False)
        if new_df.empty:
            logging.info(f"[{name}] Already updated")
            return old_df

        new_df.columns = new_df.columns.get_level_values(0)
        new_df = new_df.reset_index()
        new_df["Ticker"] = name

        updated_df = (pd.concat([old_df, new_df]).drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True))
        logging.info(f"[{name}] +{len(new_df)} new lines")
        return updated_df

    except Exception as e:
        logging.error(f"[{name}] Error updating: {e}")
        return None
    
def save_market_data(name: str, df: pd.DataFrame, file_path: Path) -> None:
    """Saving the individual CSV file"""
    df.to_csv(file_path, index=False)
    logging.info(f"[{name}] Saved in {file_path}")


def main() -> None:
    setup_logging()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    all_dfs: list[pd.DataFrame] = []

    for name, symbol in TICKERS.items():
        file_path = RAW_DIR / f"{name}.csv"

        if file_path.exists():
            df = update_ticker(name, symbol, file_path)
        else:
            df = download_ticker(name, symbol)

        if df is None:
            logging.warning(f"[{name}] Ignored (data unavailable)")
            continue

        save_market_data(name, df, file_path)
        all_dfs.append(df)

    if not all_dfs:
        logging.error("No data fetched — market_data.csv was not created")
        return

    market_data = (pd.concat(all_dfs).sort_values(["Ticker", "Date"]).reset_index(drop=True)    )
    out_path = RAW_DIR / "market_data.csv"
    market_data.to_csv(out_path, index=False)
    logging.info(f"market_data.csv : {len(market_data)} lines, {market_data['Ticker'].nunique()} tickers")


if __name__ == "__main__":
    main()