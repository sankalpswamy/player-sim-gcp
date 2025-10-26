import numpy as np, joblib, pandas as pd
import os

class SimilarityEngine:
    def __init__(self, parq_path, preproc_path, index_path=None):
        self.players = pd.read_parquet(parq_path)
        pack = joblib.load(preproc_path)
        self.scaler = pack["scaler"]
        self.feature_cols = pack["feature_cols"]
        X = self.players[self.feature_cols].fillna(0.0).values
        Xz = self.scaler.transform(X)
        self.Xn = Xz / (np.linalg.norm(Xz, axis=1, keepdims=True) + 1e-9)
        self.index = None
        self.use_faiss = False
        if index_path and os.path.exists(index_path):
            try:
                import faiss
                self.index = faiss.read_index(index_path)
                self.use_faiss = True
            except Exception:
                self.index = None

    def query(self, inputs: dict, k: int = 5):
        x = np.array([[inputs.get(c, 0.0) for c in self.feature_cols]])
        xz = self.scaler.transform(x)
        xn = xz / (np.linalg.norm(xz, axis=1, keepdims=True) + 1e-9)

        if self.index is not None and self.use_faiss:
            import faiss
            D, I = self.index.search(xn.astype("float32"), k)
            sims = D[0]
            idxs = I[0]
        else:
            sims = (self.Xn @ xn.T).ravel()
            idxs = np.argsort(-sims)[:k]

        results = self.players.iloc[idxs][["player_id","player_name","season","position"]].copy()
        results["similarity"] = sims[idxs]
        return results.to_dict(orient="records")