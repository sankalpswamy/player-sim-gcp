import os
import argparse
import pandas as pd
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler
import joblib

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT") or "YOUR_GCP_PROJECT"

MODES = {
    "offense": {
        "view": "football.features_offense",
        "cols": [
            "passing_yards_pg","passing_tds_pg","ints_pg",
            "rushing_yards_pg","rushing_tds_pg","receiving_yards_pg"
        ]
    },
    "defense": {
        "view": "football.features_defense",
        "cols": [
            "tackles_pg","sacks_pg","ints_pg","ff_pg","fr_pg","pd_pg","tfl_pg"
        ]
    }
}

def load_bq(view):
    client = bigquery.Client(project=PROJECT)
    return client.query(f"SELECT * FROM `{PROJECT}.{view}`").to_dataframe()


def build_and_save(mode: str, out_root: str = "artifacts"):
    spec = MODES[mode]
    out_dir = os.path.join(out_root, mode)
    os.makedirs(out_dir, exist_ok=True)

    df = load_bq(spec["view"])
    feature_cols = spec["cols"]

    # Fit scaler on current table
    X = df[feature_cols].fillna(0.0).values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    # Save artifacts
    df.to_parquet(f"{out_dir}/players.parquet", index=False)
    joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, f"{out_dir}/preproc.joblib")

    # Optional FAISS
    try:
        import numpy as np
        import faiss
        Xn = Xz / (np.linalg.norm(Xz, axis=1, keepdims=True) + 1e-9)
        index = faiss.IndexFlatIP(Xn.shape[1])
        faiss.normalize_L2(Xn)
        index.add(Xn.astype("float32"))
        faiss.write_index(index, f"{out_dir}/faiss.index")
        print(f"[{mode}] FAISS index written")
    except Exception as e:
        print(f"[{mode}] FAISS not available; using numpy at runtime.", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES.keys()), default="offense")
    parser.add_argument("--out_root", default="artifacts")
    args = parser.parse_args()
    build_and_save(args.mode, args.out_root)