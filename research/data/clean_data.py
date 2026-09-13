from pathlib import Path
import pandas as pd

def clean_file(path, output_dir="data/processed"):
    path = Path(path)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()
    outdir.joinpath(path.name).write_text(df.to_csv())
    return df

def clean_all(raw_dir="data/raw", output_dir="data/processed"):
    for p in Path(raw_dir).glob("*.csv"):
        clean_file(p, output_dir)
