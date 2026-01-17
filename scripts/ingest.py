import os
import glob
import uuid
import json
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "heritage_transcripts")
VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))
DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DATA_DIR = os.path.join(os.getcwd(), "data", "transcripts")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

HEADERS_JSON = {"Content-Type": "application/json"}

# Setup session with connection pooling and retries
def create_session() -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

SESSION = create_session()


def ensure_collection():
    """Check if collection exists, create if not."""
    try:
        r = SESSION.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=10)
        if r.status_code == 200:
            print(f"✓ Collection '{COLLECTION}' already exists.")
            return
    except requests.exceptions.RequestException as e:
        print(f"Error checking collection: {e}")
        raise
    
    # Create collection
    payload = {
        "vectors": {
            "size": VECTOR_SIZE,
            "distance": DISTANCE
        },
        "optimizers_config": {
            "indexing_threshold": 20000
        },
        "hnsw_config": {
            "m": 16,
            "ef_construct": 100
        }
    }
    r = SESSION.put(f"{QDRANT_URL}/collections/{COLLECTION}", headers=HEADERS_JSON, 
                    data=json.dumps(payload), timeout=10)
    r.raise_for_status()
    print(f"✓ Created collection '{COLLECTION}' with optimized HNSW config.")


def chunk_text(text: str, max_chars: int = 600, overlap: int = 50) -> List[str]:
    """Chunk text into paragraphs with optional overlap for better context."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = []
    total = 0
    
    for p in paras:
        p_len = len(p)
        if total + p_len + 2 <= max_chars:
            buf.append(p)
            total += p_len + 2
        else:
            if buf:
                chunk = "\n\n".join(buf)
                chunks.append(chunk)
                # Keep last paragraph for overlap if it's small enough
                if overlap > 0 and len(buf[-1]) <= overlap:
                    buf = [buf[-1], p]
                    total = len(buf[-1]) + p_len + 2
                else:
                    buf = [p]
                    total = p_len
            else:
                # Single paragraph too large, split by sentences
                chunks.append(p)
                buf = []
                total = 0
    if buf:
        chunks.append("\n\n".join(buf))
    
    return [c for c in chunks if c.strip()]


def embed(text: str) -> List[float]:
    """Generate embedding for a single text."""
    try:
        payload = {"model": EMBED_MODEL, "prompt": text}
        r = SESSION.post(f"{OLLAMA_URL}/api/embeddings", headers=HEADERS_JSON, 
                        data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("embedding")
    except requests.exceptions.RequestException as e:
        print(f"Error embedding text: {e}")
        raise

def embed_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings in parallel for better throughput."""
    embeddings = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(embed, text): i for i, text in enumerate(texts)}
        for future in as_completed(future_to_idx):
            try:
                embeddings.append((future_to_idx[future], future.result()))
            except Exception as e:
                idx = future_to_idx[future]
                print(f"Failed to embed chunk {idx}: {e}")
                embeddings.append((idx, None))
    # Sort by original index
    embeddings.sort(key=lambda x: x[0])
    return [emb for _, emb in embeddings]


def upsert_points(points: List[Dict[str, Any]], batch_size: int = BATCH_SIZE):
    """Upsert points in batches for better performance."""
    if not points:
        return
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        payload = {"points": batch}
        try:
            r = SESSION.put(f"{QDRANT_URL}/collections/{COLLECTION}/points", 
                          headers=HEADERS_JSON, data=json.dumps(payload), timeout=30)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error upserting batch {i//batch_size + 1}: {e}")
            raise


def ingest_file(path: str) -> int:
    """Ingest a single file with optimized batch embedding."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return 0
    
    # Parse front-matter metadata
    meta = {}
    if content.startswith("---"):
        fm_end = content.find("---", 3)
        if fm_end != -1:
            fm = content[3:fm_end].strip()
            for line in fm.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"').strip("'")
            body = content[fm_end+3:].strip()
        else:
            body = content
    else:
        body = content
    
    chunks = chunk_text(body)
    if not chunks:
        print(f"⚠ No chunks extracted from {os.path.basename(path)}")
        return 0
    
    # Batch embed all chunks at once
    embeddings = embed_batch(chunks)
    
    # Build points
    points = []
    for idx, (ch, vec) in enumerate(zip(chunks, embeddings)):
        if vec is None:
            continue
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
    
    if points:
        upsert_points(points)
    
    return len(points)


def main():
    """Main ingestion pipeline with optimized parallel processing."""
    start_time = time.time()
    
    print(f"🚀 Starting Tunisian Heritage ingestion...")
    print(f"📂 Data directory: {DATA_DIR}")
    print(f"🔗 Ollama: {OLLAMA_URL}")
    print(f"🔗 Qdrant: {QDRANT_URL}")
    print(f"⚙️  Batch size: {BATCH_SIZE}, Workers: {MAX_WORKERS}\n")
    
    ensure_collection()
    
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.md")))
    if not files:
        print(f"❌ No files found in {DATA_DIR}")
        return
    
    print(f"📄 Found {len(files)} files to ingest\n")
    
    total_chunks = 0
    failed_files = []
    
    # Process files with progress bar
    iterator = tqdm(files, desc="Processing files") if tqdm else files
    
    for fp in iterator:
        try:
            chunks = ingest_file(fp)
            total_chunks += chunks
            if not tqdm:
                print(f"✓ {os.path.basename(fp)}: {chunks} chunks")
        except Exception as e:
            failed_files.append((fp, str(e)))
            if not tqdm:
                print(f"✗ {os.path.basename(fp)}: FAILED - {e}")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"✅ Ingestion complete!")
    print(f"📊 Total chunks: {total_chunks}")
    print(f"⏱️  Time elapsed: {elapsed:.2f}s")
    print(f"⚡ Throughput: {total_chunks/elapsed:.1f} chunks/sec")
    
    if failed_files:
        print(f"\n⚠️  Failed files ({len(failed_files)}):")
        for fp, err in failed_files:
            print(f"  - {os.path.basename(fp)}: {err}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
