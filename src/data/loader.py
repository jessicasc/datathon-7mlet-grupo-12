from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

KAGGLE_DATASET = ROOT_DIR / "data" / "kaggle" / "selected-dataset.csv"
PROCESSED_DATASET = ROOT_DIR / "data" / "processed" / "modeling_table.parquet"

# carrega a base original do Kaggle sem alterações
def load_raw_dataset() -> pd.DataFrame:

    return pd.read_csv(KAGGLE_DATASET, sep=";")

# carrega a base tratada
def load_processed_dataset() -> pd.DataFrame:

    return pd.read_parquet(PROCESSED_DATASET)