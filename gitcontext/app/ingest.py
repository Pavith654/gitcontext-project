"""Phase 1-2: read repo files + commits, chunk, embed, store in Chroma."""
import ast
import re
from pathlib import Path

import chromadb
from git import Repo
from sentence_transformers import SentenceTransformer

LANGS = {".py": "python", ".js": "javascript", ".ts": "typescript",
         ".java": "java", ".md": "markdown", ".sql": "sql"}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build"}
SECRET_HINTS = re.compile(r"(\.env|\.pem|\.key|secret|credential|id_rsa)", re.I)
MAX_BYTES = 300_000

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")


def chunk_lines(text, size=60, overlap=10):
    lines = text.splitlines()
    step = size - overlap
    for i in range(0, max(len(lines), 1), step):
        part = "\n".join(lines[i:i + size])
        if part.strip():
            yield part, i + 1, min(i + size, len(lines)), ""


def chunk_python(text):
    """Syntax-aware: one chunk per top-level function/class."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        yield from chunk_lines(text)
        return
    found = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seg = ast.get_source_segment(text, node)
            if not seg:
                continue
            found = True
            if len(seg) > 3000:  # very large -> split but keep the name
                for part, s, e, _ in chunk_lines(seg):
                    yield part, node.lineno + s - 1, node.lineno + e - 1, node.name
            else:
                yield seg, node.lineno, node.end_lineno, node.name
    if not found:
        yield from chunk_lines(text)


def load_files(repo_path):
    for p in Path(repo_path).rglob("*"):
        if not p.is_file() or p.suffix not in LANGS:
            continue
        if any(part in SKIP_DIRS for part in p.parts) or SECRET_HINTS.search(p.name):
            continue
        if p.stat().st_size > MAX_BYTES:
            continue
        try:
            yield p, p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def load_commits(repo, max_count=200):
    for c in repo.iter_commits(max_count=max_count):
        files = list(c.stats.files.keys())
        diff_txt = ""
        if c.parents:
            for d in c.parents[0].diff(c, create_patch=True):
                diff_txt += d.diff.decode("utf-8", errors="ignore")[:800]
                if len(diff_txt) > 2000:
                    break
        text = (f"Commit {c.hexsha[:7]} by {c.author.name} on {c.committed_datetime:%Y-%m-%d}\n"
                f"Message: {c.message.strip()}\nFiles: {', '.join(files)}\nDiff:\n{diff_txt[:2000]}")
        yield c.hexsha, text


def index_repo(repo_path, name):
    repo = Repo(repo_path)
    head = repo.head.commit.hexsha
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name)

    ids, docs, metas = [], [], []
    root = Path(repo_path)
    for p, text in load_files(repo_path):
        rel = str(p.relative_to(root))
        lang = LANGS[p.suffix]
        chunker = chunk_python(text) if lang == "python" else chunk_lines(text)
        for i, (chunk, s, e, fn) in enumerate(chunker):
            ids.append(f"file:{rel}:{i}")
            docs.append(f"File: {rel}\n{chunk}")
            metas.append({"type": "source_code" if lang != "markdown" else "docs",
                          "file_path": rel, "language": lang, "function_name": fn,
                          "start_line": s, "end_line": e, "indexed_sha": head})
    for sha, text in load_commits(repo):
        ids.append(f"commit:{sha}")
        docs.append(text)
        metas.append({"type": "commit", "file_path": "", "language": "", "function_name": "",
                      "start_line": 0, "end_line": 0, "indexed_sha": sha})

    for i in range(0, len(ids), 64):
        batch = docs[i:i + 64]
        col.add(ids=ids[i:i + 64], documents=batch, metadatas=metas[i:i + 64],
                embeddings=model.encode(batch).tolist())
    return {"chunks": len(ids), "indexed_sha": head}
