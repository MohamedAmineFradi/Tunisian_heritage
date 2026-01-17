"""
Optimized Query Script for Tunisian Heritage RAG
Supports caching, batch queries, and direct API access.
"""
import os
import sys
import json
import argparse
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv

# Import cache if available
try:
    from cache import get_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    print("Warning: cache.py not found, caching disabled")

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "heritage_transcripts")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.getenv("GEN_MODEL", "mixtral")
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"

HEADERS_JSON = {"Content-Type": "application/json"}


def embed_query(query: str, use_cache: bool = True) -> List[float]:
    """
    Generate embedding for a query with optional caching.
    
    Args:
        query: Query text
        use_cache: Whether to use cache
        
    Returns:
        Embedding vector
    """
    # Try cache first
    if use_cache and CACHE_AVAILABLE and ENABLE_CACHE:
        cache = get_cache()
        cached_emb = cache.get_embedding(query, EMBED_MODEL)
        if cached_emb:
            print(f"✓ Using cached embedding")
            return cached_emb
    
    # Generate embedding
    payload = {"model": EMBED_MODEL, "prompt": query}
    r = requests.post(f"{OLLAMA_URL}/api/embeddings", headers=HEADERS_JSON, 
                     data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    embedding = r.json().get("embedding")
    
    # Cache for future use
    if use_cache and CACHE_AVAILABLE and ENABLE_CACHE and embedding:
        cache = get_cache()
        cache.set_embedding(query, embedding, EMBED_MODEL)
    
    return embedding


def search_qdrant(embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search Qdrant for similar documents.
    
    Args:
        embedding: Query embedding
        limit: Number of results
        
    Returns:
        List of search results with payload and score
    """
    payload = {
        "vector": embedding,
        "limit": limit,
        "with_payload": True,
        "with_vector": False
    }
    r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
                     headers=HEADERS_JSON, data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    return r.json().get("result", [])


def generate_answer(query: str, context: str) -> str:
    """
    Generate answer using LLM with provided context.
    
    Args:
        query: User query
        context: Retrieved context from vector search
        
    Returns:
        Generated answer
    """
    system_prompt = (
        "You are a historian assistant for Tunisian heritage. "
        "Use provided context verbatim when citing. "
        "Answer in Arabic or French. Include brief citations with source labels [#]."
    )
    
    prompt = f"""Question: {query}

Context (citations):
{context}

Write a concise answer, include citation snippets with source labels [#]."""
    
    payload = {
        "model": GEN_MODEL,
        "stream": False,
        "system": system_prompt,
        "prompt": prompt
    }
    
    r = requests.post(f"{OLLAMA_URL}/api/generate", headers=HEADERS_JSON,
                     data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    return r.json().get("response", "")


def format_context(results: List[Dict[str, Any]]) -> str:
    """
    Format search results into context string.
    
    Args:
        results: Search results from Qdrant
        
    Returns:
        Formatted context string
    """
    context_parts = []
    for idx, result in enumerate(results, 1):
        payload = result.get("payload", {})
        text = payload.get("text", "")
        title = payload.get("title", "Unknown")
        source = payload.get("file", "Unknown")
        score = result.get("score", 0)
        
        context_parts.append(
            f"[#{idx} | {title} | {source} | score: {score:.3f}]\n{text}"
        )
    
    return "\n\n".join(context_parts)


def query_rag(query: str, limit: int = 5, use_cache: bool = True, 
              generate: bool = True) -> Dict[str, Any]:
    """
    Complete RAG query pipeline.
    
    Args:
        query: User query
        limit: Number of search results
        use_cache: Whether to use caching
        generate: Whether to generate answer (False for search only)
        
    Returns:
        Dictionary with results and answer
    """
    print(f"\n🔍 Query: {query}")
    print(f"{'='*60}")
    
    # Embed query
    print(f"\n⚙️  Generating embedding...")
    embedding = embed_query(query, use_cache=use_cache)
    
    # Search
    print(f"🔎 Searching Qdrant (top {limit})...")
    results = search_qdrant(embedding, limit=limit)
    
    print(f"\n📚 Found {len(results)} results:")
    for idx, result in enumerate(results, 1):
        payload = result.get("payload", {})
        title = payload.get("title", "Unknown")
        score = result.get("score", 0)
        print(f"  {idx}. {title} (score: {score:.3f})")
    
    response = {
        "query": query,
        "results": results,
        "answer": None
    }
    
    # Generate answer
    if generate and results:
        print(f"\n🤖 Generating answer with {GEN_MODEL}...")
        context = format_context(results)
        answer = generate_answer(query, context)
        response["answer"] = answer
        
        print(f"\n💬 Answer:\n{'-'*60}")
        print(answer)
        print(f"{'-'*60}")
    
    return response


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Query Tunisian Heritage RAG")
    parser.add_argument("query", nargs="?", help="Query text")
    parser.add_argument("-l", "--limit", type=int, default=5, 
                       help="Number of search results (default: 5)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching")
    parser.add_argument("--search-only", action="store_true",
                       help="Only search, don't generate answer")
    parser.add_argument("-i", "--interactive", action="store_true",
                       help="Interactive mode")
    parser.add_argument("--cache-stats", action="store_true",
                       help="Show cache statistics")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear cache")
    
    args = parser.parse_args()
    
    # Cache operations
    if args.cache_stats and CACHE_AVAILABLE:
        cache = get_cache()
        stats = cache.get_stats()
        print(f"\n📊 Cache Statistics:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Active entries: {stats['active_entries']}")
        print(f"  Expired entries: {stats['expired_entries']}")
        print(f"  Cache file: {stats['cache_file']}")
        print(f"  TTL: {stats['ttl']}s")
        return
    
    if args.clear_cache and CACHE_AVAILABLE:
        cache = get_cache()
        cache.clear()
        print("✓ Cache cleared")
        return
    
    # Interactive mode
    if args.interactive:
        print("🎯 Interactive Query Mode (type 'quit' to exit)\n")
        while True:
            try:
                query_text = input("Query: ").strip()
                if query_text.lower() in ('quit', 'exit', 'q'):
                    break
                if not query_text:
                    continue
                
                query_rag(
                    query_text,
                    limit=args.limit,
                    use_cache=not args.no_cache,
                    generate=not args.search_only
                )
                print()
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        return
    
    # Single query mode
    if not args.query:
        parser.print_help()
        return
    
    try:
        query_rag(
            args.query,
            limit=args.limit,
            use_cache=not args.no_cache,
            generate=not args.search_only
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
