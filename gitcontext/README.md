# GitContext (starter)

RAG assistant for a Git repository: code (function-level chunks), docs, and commit history.

## Run
```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key                    # Windows: set ANTHROPIC_API_KEY=your_key
uvicorn app.main:app --reload                        # terminal 1
streamlit run ui.py                                  # terminal 2
```
In the sidebar, enter the path of a cloned repo, click **Index repository**, then ask questions.

## Roadmap
- [x] Code/doc chunking, metadata, embeddings, Chroma
- [x] Hybrid search (vector + BM25, RRF), optional cross-encoder rerank
- [x] Grounded answers with numbered source citations
- [x] Commit history indexing (local git)
- [ ] GitHub API: PRs, issues, discussions
- [ ] Incremental re-indexing by commit SHA
- [ ] Call-graph / import relationships
- [ ] Evaluation set (question -> expected file/commit)
- [ ] Auth, per-repo access control, secret scanning
