import numpy as np
import pandas as pd
from hmmlearn import GaussianHMM
import logging
from pathlib import Path
import joblib

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models/clustering")

def detect_regimes(features_path: Path) -> pd.DataFrame:
    """Apply the HMM model to the data"""


def main():
    logging.basicConfig(level=logging.INFO)
