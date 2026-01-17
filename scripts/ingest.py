import os
import glob
import uuid
import json
import time
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "heritage_transcripts")
VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))
DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DATA_DIR = os.path.join(os.getcwd(), "data", "transcripts")

HEADERS_JSON = {"Content-Type": "application/json"}


def ensure_collection():
    # Check exists
    r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}")
    if r.status_code == 200:
        print(f"Collection '{COLLECTION}' already exists.")
        return
    # Create collection
    payload = {
        "vectors": {
            "size": VECTOR_SIZE,
            "distance": DISTANCE
        }
    }
    r = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}", headers=HEADERS_JSON, data=json.dumps(payload))
    r.raise_for_status()
    print(f"Created collection '{COLLECTION}'.")


def chunk_text(text: str, max_chars: int = 600) -> List[str]:
    # Simple paragraph splitter and rejoin small ones
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = []
    total = 0
    for p in paras:
        if total + len(p) + 2 <= max_chars:
            buf.append(p)
            total += len(p) + 2
        else:
            if buf:
                chunks.append("\n\n".join(buf))
            buf = [p]
            total = len(p)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def embed(text: str) -> List[float]:
    payload = {"model": EMBED_MODEL, "prompt": text}
    r = requests.post(f"{OLLAMA_URL}/api/embeddings", headers=HEADERS_JSON, data=json.dumps(payload))
    r.raise_for_status()
    data = r.json()
    return data.get("embedding")


def upsert_points(points: List[dict]):
    payload = {"points": points}
    r = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}/points", headers=HEADERS_JSON, data=json.dumps(payload))
    r.raise_for_status()


def ingest_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # crude front-matter parsing
    meta = {}
    if content.startswith("---"):
        fm_end = content.find("---", 3)
        fm = content[3:fm_end].strip()
        for line in fm.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
        body = content[fm_end+3:].strip()
    else:
        body = content
    chunks = chunk_text(body)
    points = []
    for idx, ch in enumerate(chunks):
        vec = embed(ch)
        point_id = str(uuid.uuid4())
        payload = {
            "source": meta.get("source", "simulated"),
            "title": meta.get("title", os.path.basename(path)),
            "region": meta.get("region", None),
            "date": meta.get("date", None),
            "lang": meta.get("lang", None),
            "chunk_index": idx,
            "text": ch,
            "file": os.path.basename(path)
        }
        points.append({"id": point_id, "vector": vec, "payload": payload})
        # throttle a bit
        time.sleep(0.05)
    upsert_points(points)
    print(f"Ingested {len(points)} chunks from {os.path.basename(path)}")


def main():
    ensure_collection()
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.md")))
    if not files:
        print(f"No files found in {DATA_DIR}")
        return
    for fp in files:
        ingest_file(fp)
    print("Done.")


if __name__ == "__main__":
    main()
