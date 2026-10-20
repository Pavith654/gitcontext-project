from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .ingest import client, index_repo
from .llm import answer
from .retrieval import clear_cache, hybrid_search

app = FastAPI(title="GitContext")


class IndexReq(BaseModel):
    name: str
    path: str  # local path to a cloned git repo


class AskReq(BaseModel):
    repository_id: str
    question: str
    rerank: bool = False


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/repositories")
def add_repo(req: IndexReq):
    try:
        info = index_repo(req.path, req.name)
    except Exception as e:
        raise HTTPException(400, str(e))
    clear_cache(req.name)
    return info


@app.get("/repositories")
def list_repos():
    return [c.name for c in client.list_collections()]


@app.post("/ask")
def ask(req: AskReq):
    try:
        results = hybrid_search(req.repository_id, req.question, rerank=req.rerank)
    except Exception:
        raise HTTPException(404, "Repository not indexed")
    text, sources = answer(req.question, results)
    return {"answer": text, "sources": sources}
