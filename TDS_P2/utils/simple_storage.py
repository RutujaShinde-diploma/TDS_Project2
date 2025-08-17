"""
Simple in-memory storage system to replace Redis
Perfect for university projects and demos
"""

import time
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class SimpleStorage:
    """Simple in-memory storage with file persistence"""
    
    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # In-memory storage
        self.cache: Dict[str, Any] = {}
        self.jobs: Dict[str, Any] = {}
        
        # Load existing data from files
        self._load_from_files()
    
    def _load_from_files(self):
        """Load existing data from storage files"""
        try:
            # Load cache
            cache_file = self.storage_dir / "cache.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached items")
            
            # Load jobs
            jobs_file = self.storage_dir / "jobs.json"
            if jobs_file.exists():
                with open(jobs_file, 'r') as f:
                    self.jobs = json.load(f)
                logger.info(f"Loaded {len(self.jobs)} jobs")
                
        except Exception as e:
            logger.warning(f"Failed to load storage files: {e}")
    
    def _save_to_files(self):
        """Save data to storage files"""
        try:
            # Save cache
            cache_file = self.storage_dir / "cache.json"
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            
            # Save jobs
            jobs_file = self.storage_dir / "jobs.json"
            with open(jobs_file, 'w') as f:
                json.dump(self.jobs, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save storage files: {e}")
    
    # Cache methods
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set a cache value with TTL"""
        try:
            self.cache[key] = {
                "value": value,
                "expires": time.time() + ttl
            }
            self._save_to_files()
            return True
        except Exception as e:
            logger.error(f"Failed to set cache: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get a cache value if not expired"""
        try:
            if key in self.cache:
                item = self.cache[key]
                if time.time() < item["expires"]:
                    return item["value"]
                else:
                    # Expired, remove it
                    del self.cache[key]
                    self._save_to_files()
            return None
        except Exception as e:
            logger.error(f"Failed to get cache: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a cache key"""
        try:
            if key in self.cache:
                del self.cache[key]
                self._save_to_files()
            return True
        except Exception as e:
            logger.error(f"Failed to delete cache: {e}")
            return False
    
    def search_keys(self, pattern: str) -> List[str]:
        """Search for cache keys matching pattern"""
        try:
            if "*" in pattern:
                # Simple wildcard search
                prefix = pattern.split("*")[0]
                return [key for key in self.cache.keys() if key.startswith(prefix)]
            else:
                return [key for key in self.cache.keys() if pattern in key]
        except Exception as e:
            logger.error(f"Failed to search keys: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            # Clean expired items first
            current_time = time.time()
            expired_keys = [
                key for key, item in self.cache.items() 
                if current_time >= item["expires"]
            ]
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                self._save_to_files()
            
            return {
                "total_cache_items": len(self.cache),
                "total_jobs": len(self.jobs),
                "storage_dir": str(self.storage_dir.absolute())
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def is_connected(self) -> bool:
        """Always return True for simple storage"""
        return True
    
    # Job storage methods
    def store_job(self, job_id: str, job_data: Dict[str, Any]) -> bool:
        """Store a job"""
        try:
            self.jobs[job_id] = {
                **job_data,
                "stored_at": time.time()
            }
            self._save_to_files()
            return True
        except Exception as e:
            logger.error(f"Failed to store job: {e}")
            return False
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID"""
        return self.jobs.get(job_id)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for jobs"""
        return {
            "total_jobs": len(self.jobs),
            "storage_dir": str(self.storage_dir.absolute())
        }

# Global storage instance
simple_storage = SimpleStorage()
