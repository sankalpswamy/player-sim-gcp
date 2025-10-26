import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .similarity import SimilarityEngine

app = FastAPI(title="Player Similarity API")

_engines = {}

def get_engine(mode: str):
    if mode not in _engines:
        base = os.getenv("ARTIFACT_DIR", "/app/artifacts")
        mode_dir = os.path.join(base, mode)
        parq = os.path.join(mode_dir, "players.parquet")
        preproc = os.path.join(mode_dir, "preproc.joblib")
        indexp = os.path.join(mode_dir, "faiss.index")
        _engines[mode] = SimilarityEngine(parq, preproc, indexp)
    return _engines[mode]

class OffenseInput(BaseModel):
    passing_yards_pg: float = Field(ge=0)
    passing_tds_pg: float = Field(ge=0)
    ints_pg: float = Field(ge=0)
    rushing_yards_pg: float = Field(ge=0)
    rushing_tds_pg: float = Field(ge=0)
    receiving_yards_pg: float = Field(ge=0)
    k: int = 5

class DefenseInput(BaseModel):
    tackles_pg: float = Field(ge=0)
    sacks_pg: float = Field(ge=0)
    ints_pg: float = Field(ge=0)
    ff_pg: float = Field(ge=0)
    fr_pg: float = Field(ge=0)
    pd_pg: float = Field(ge=0)
    tfl_pg: float = Field(ge=0)
    k: int = 5

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/similarity/offense")
async def offense_sim(inp: OffenseInput):
    eng = get_engine("offense")
    recs = eng.query(inp.dict(), k=inp.k)
    return {"results": recs}

@app.post("/similarity/defense")
async def defense_sim(inp: DefenseInput):
    eng = get_engine("defense")
    recs = eng.query(inp.dict(), k=inp.k)
    return {"results": recs}