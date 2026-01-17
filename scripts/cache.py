"""
Query Cache Module for Tunisian Heritage RAG
Caches embeddings and search results to reduce API calls and improve response time.
"""
import os
import json
import hashlib
import time
from typing import Optional, Dict, Any, List
from pathlib import Path


class QueryCache:
    """Simple file-based cache for query embeddings and results."""
    
    def __init__(self, cache_file: str = ".cache/query_cache.json", ttl: int = 3600):
        """
        Initialize cache.
        
        Args:
            cache_file: Path to cache file
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.cache_file = Path(cache_file)
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                # Clean expired entries on load
                self._clean_expired()
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _clean_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, value in self.cache.items()
            if current_time - value.get('timestamp', 0) > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self._save_cache()
    
    def _hash_query(self, query: str, model: str = "") -> str:
        """Generate cache key from query and model."""
        key = f"{query}:{model}".encode('utf-8')
        return hashlib.sha256(key).hexdigest()
    
    def get_embedding(self, query: str, model: str = "nomic-embed-text") -> Optional[List[float]]:
        """
        Retrieve cached embedding for a query.
        
        Args:
            query: Query text
            model: Embedding model name
            
        Returns:
            Cached embedding or None if not found/expired
        """
        cache_key = self._hash_query(query, model)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            # Check if expired
            if time.time() - entry.get('timestamp', 0) <= self.ttl:
                return entry.get('embedding')
            else:
                # Remove expired entry
                del self.cache[cache_key]
        
        return None
    
    def set_embedding(self, query: str, embedding: List[float], model: str = "nomic-embed-text"):
        """
        Cache an embedding for a query.
        
        Args:
            query: Query text
            embedding: Embedding vector
            model: Embedding model name
        """
        cache_key = self._hash_query(query, model)
        self.cache[cache_key] = {
            'query': query,
            'model': model,
            'embedding': embedding,
            'timestamp': time.time()
        }
        self._save_cache()
    
    def get_search_results(self, query: str, limit: int = 5) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached search results.
        
        Args:
            query: Query text
            limit: Number of results
            
        Returns:
            Cached search results or None if not found/expired
        """
        cache_key = self._hash_query(f"search:{query}:{limit}", "")
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            # Check if expired
            if time.time() - entry.get('timestamp', 0) <= self.ttl:
                return entry.get('results')
            else:
                # Remove expired entry
                del self.cache[cache_key]
        
        return None
    
    def set_search_results(self, query: str, results: List[Dict[str, Any]], limit: int = 5):
        """
        Cache search results.
        
        Args:
            query: Query text
            results: Search results
            limit: Number of results
        """
        cache_key = self._hash_query(f"search:{query}:{limit}", "")
        self.cache[cache_key] = {
            'query': query,
            'limit': limit,
            'results': results,
            'timestamp': time.time()
        }
        self._save_cache()
    
    def clear(self):
        """Clear all cache entries."""
        self.cache = {}
        self._save_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        total_entries = len(self.cache)
        expired_entries = sum(
            1 for entry in self.cache.values()
            if current_time - entry.get('timestamp', 0) > self.ttl
        )
        
        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired_entries,
            'expired_entries': expired_entries,
            'cache_file': str(self.cache_file),
            'ttl': self.ttl
        }


# Global cache instance
_cache_instance: Optional[QueryCache] = None


def get_cache(cache_file: str = None, ttl: int = None) -> QueryCache:
    """
    Get or create global cache instance.
    
    Args:
        cache_file: Path to cache file (uses env var if not provided)
        ttl: Time to live in seconds (uses env var if not provided)
        
    Returns:
        QueryCache instance
    """
    global _cache_instance
    
    if _cache_instance is None:
        cache_file = cache_file or os.getenv('CACHE_FILE', '.cache/query_cache.json')
        ttl = ttl or int(os.getenv('CACHE_TTL', '3600'))
        _cache_instance = QueryCache(cache_file, ttl)
    
    return _cache_instance


if __name__ == "__main__":
    # Example usage
    cache = QueryCache()
    
    # Test embedding cache
    test_query = "Tell me about Tunisian resistance"
    test_embedding = [0.1, 0.2, 0.3] * 256  # Mock embedding
    
    cache.set_embedding(test_query, test_embedding)
    retrieved = cache.get_embedding(test_query)
    
    print(f"Cache test: {'PASSED' if retrieved == test_embedding else 'FAILED'}")
    print(f"Stats: {cache.get_stats()}")
