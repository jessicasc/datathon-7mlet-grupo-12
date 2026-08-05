from pathlib import Path
from src.data.loader import load_raw_dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "modeling_table.parquet"

FEATURE_COLUMNS = [
    "customer_id",
    "age",
    "job",
    "education",
    "marital",
    "previous",
    "poutcome",
    "y",
]

# gera a modeling_table.parquet com os tratamentos e colunas necessárias
def create_modeling_table():

    df = load_raw_dataset()

    #remove duplicatas
    df = df.drop_duplicates().reset_index(drop=True)

    # cria identificador sintético
    df.insert(0, "customer_id", range(1, len(df) + 1))

    # mantém somente as colunas selecionadas
    df = df[FEATURE_COLUMNS]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Modeling table salva em:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    create_modeling_table()