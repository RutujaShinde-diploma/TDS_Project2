import redis.asyncio as redis
import json
import logging
import time
from typing import Any, Optional, Dict, List
import hashlib

from config import config

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.fallback_cache = {}  # In-memory fallback
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            # Check if using memory fallback
            if config.REDIS_URL.startswith("memory://"):
                logger.info("Using in-memory cache (no Redis)")
                self.redis_client = None
                return
            
            self.redis_client = redis.from_url(config.REDIS_URL)
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory cache: {str(e)}")
            self.redis_client = None
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        return self.redis_client is not None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                # Use fallback cache
                cache_entry = self.fallback_cache.get(key)
                if cache_entry:
                    if time.time() < cache_entry['expires']:
                        return cache_entry['value']
                    else:
                        del self.fallback_cache[key]
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or config.CACHE_TTL
            
            if self.redis_client:
                serialized = json.dumps(value, default=str)
                await self.redis_client.setex(key, ttl, serialized)
                return True
            else:
                # Use fallback cache
                self.fallback_cache[key] = {
                    'value': value,
                    'expires': time.time() + ttl
                }
                
                # Clean up expired entries periodically
                if len(self.fallback_cache) > 100:
                    self._cleanup_fallback_cache()
                
                return True
                
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if self.redis_client:
                await self.redis_client.delete(key)
            else:
                self.fallback_cache.pop(key, None)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if self.redis_client:
                return await self.redis_client.exists(key) > 0
            else:
                cache_entry = self.fallback_cache.get(key)
                if cache_entry and time.time() < cache_entry['expires']:
                    return True
                elif cache_entry:
                    del self.fallback_cache[key]
                
                return False
                
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {str(e)}")
            return False
    
    def _cleanup_fallback_cache(self):
        """Clean up expired entries from fallback cache"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.fallback_cache.items()
            if current_time >= entry['expires']
        ]
        
        for key in expired_keys:
            del self.fallback_cache[key]
    
    def generate_cache_key(self, prefix: str, *args) -> str:
        """Generate a consistent cache key"""
        key_parts = [prefix] + [str(arg) for arg in args]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and debug information"""
        try:
            stats = {
                "redis_connected": self.is_connected(),
                "fallback_cache_size": len(self.fallback_cache),
                "timestamp": time.time()
            }
            
            if self.redis_client:
                try:
                    # Get Redis info
                    info = await self.redis_client.info()
                    stats.update({
                        "redis_keys": info.get('db0', {}).get('keys', 0),
                        "redis_memory": info.get('used_memory_human', 'unknown'),
                        "redis_uptime": info.get('uptime_in_seconds', 0)
                    })
                except Exception as e:
                    stats["redis_info_error"] = str(e)
            
            return stats
            
        except Exception as e:
            return {"error": str(e), "timestamp": time.time()}
    
    async def search_cache_keys(self, pattern: str = "*") -> List[str]:
        """Search for cache keys matching a pattern (Redis only)"""
        try:
            if not self.redis_client:
                return []
            
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key.decode())
            
            return keys
            
        except Exception as e:
            logger.error(f"Error searching cache keys: {str(e)}")
            return []

# LLM Response Cache
class LLMCache:
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.prefix = "llm"
    
    async def get_response(self, prompt: str, model: str, temperature: float = 0.1) -> Optional[str]:
        """Get cached LLM response"""
        cache_key = self.cache_manager.generate_cache_key(
            self.prefix, prompt, model, temperature
        )
        return await self.cache_manager.get(cache_key)
    
    async def cache_response(self, prompt: str, model: str, response: str, temperature: float = 0.1) -> bool:
        """Cache LLM response"""
        cache_key = self.cache_manager.generate_cache_key(
            self.prefix, prompt, model, temperature
        )
        return await self.cache_manager.set(cache_key, response, ttl=config.CACHE_TTL)

# Code Pattern Cache
class CodeCache:
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.prefix = "code"
    
    async def get_code(self, action_type: str, context: str) -> Optional[str]:
        """Get cached code for action type and context"""
        cache_key = self.cache_manager.generate_cache_key(
            self.prefix, action_type, context
        )
        return await self.cache_manager.get(cache_key)
    
    async def cache_code(self, action_type: str, context: str, code: str) -> bool:
        """Cache working code pattern"""
        cache_key = self.cache_manager.generate_cache_key(
            self.prefix, action_type, context
        )
        return await self.cache_manager.set(cache_key, code, ttl=config.CACHE_TTL * 2)  # Longer TTL for code
