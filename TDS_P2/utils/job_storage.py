"""
Persistent job storage manager that saves job results to disk
to survive server restarts and prevent data loss.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any
import threading
from dataclasses import dataclass, asdict
from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class JobRecord:
    """Represents a complete job record with all metadata"""
    job_id: str
    status: str
    created_at: float
    updated_at: float
    execution_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    plan: Optional[Dict] = None
    workspace_path: Optional[str] = None

class PersistentJobStorage:
    """
    Manages persistent storage of job data to disk.
    Jobs are stored as individual JSON files for easy access and cleanup.
    """
    
    def __init__(self, storage_dir: str = "job_storage", cleanup_hours: int = 24):
        """
        Initialize persistent job storage.
        
        Args:
            storage_dir: Directory to store job files
            cleanup_hours: Hours after which completed jobs are cleaned up
        """
        self.storage_dir = Path(storage_dir)
        self.cleanup_hours = cleanup_hours
        self._lock = threading.RLock()  # Thread-safe operations
        
        # Create storage directory
        self.storage_dir.mkdir(exist_ok=True)
        
        # Start cleanup task
        self._start_cleanup_task()
        
        logger.info(f"Initialized persistent job storage at {self.storage_dir}")
    
    def _get_job_file(self, job_id: str) -> Path:
        """Get the file path for a job ID"""
        return self.storage_dir / f"{job_id}.json"
    
    def create_job(self, job_id: str, workspace_path: str) -> JobRecord:
        """Create a new job record"""
        with self._lock:
            current_time = time.time()
            job_record = JobRecord(
                job_id=job_id,
                status="submitted",
                created_at=current_time,
                updated_at=current_time,
                workspace_path=workspace_path
            )
            
            self._save_job(job_record)
            logger.info(f"Created job record for {job_id}")
            return job_record
    
    def update_job(self, job_id: str, **updates) -> Optional[JobRecord]:
        """Update a job record with new data"""
        with self._lock:
            job_record = self.get_job(job_id)
            if not job_record:
                logger.warning(f"Attempted to update non-existent job {job_id}")
                return None
            
            # Update fields
            for key, value in updates.items():
                if hasattr(job_record, key):
                    setattr(job_record, key, value)
            
            job_record.updated_at = time.time()
            self._save_job(job_record)
            
            logger.debug(f"Updated job {job_id}: {list(updates.keys())}")
            return job_record
    
    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Retrieve a job record by ID"""
        with self._lock:
            job_file = self._get_job_file(job_id)
            
            if not job_file.exists():
                return None
            
            try:
                with open(job_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                return JobRecord(**data)
            
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.error(f"Failed to load job {job_id}: {e}")
                return None
    
    def _save_job(self, job_record: JobRecord):
        """Save a job record to disk"""
        job_file = self._get_job_file(job_record.job_id)
        
        try:
            # Convert to dict and handle non-serializable objects
            data = asdict(job_record)
            
            # Write atomically (write to temp file, then rename)
            temp_file = job_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Atomic rename
            temp_file.replace(job_file)
            
        except Exception as e:
            logger.error(f"Failed to save job {job_record.job_id}: {e}")
            raise
    
    def list_jobs(self, status_filter: Optional[str] = None) -> Dict[str, JobRecord]:
        """List all jobs, optionally filtered by status"""
        with self._lock:
            jobs = {}
            
            for job_file in self.storage_dir.glob("*.json"):
                if job_file.stem == "cleanup_log":  # Skip internal files
                    continue
                
                job_record = self.get_job(job_file.stem)
                if job_record:
                    if not status_filter or job_record.status == status_filter:
                        jobs[job_record.job_id] = job_record
            
            return jobs
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job record"""
        with self._lock:
            job_file = self._get_job_file(job_id)
            
            if job_file.exists():
                try:
                    job_file.unlink()
                    logger.info(f"Deleted job record {job_id}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete job {job_id}: {e}")
                    return False
            
            return False
    
    def cleanup_old_jobs(self) -> int:
        """Clean up old completed/failed jobs"""
        with self._lock:
            cutoff_time = time.time() - (self.cleanup_hours * 3600)
            cleaned_count = 0
            
            for job_file in self.storage_dir.glob("*.json"):
                if job_file.stem == "cleanup_log":
                    continue
                
                job_record = self.get_job(job_file.stem)
                if not job_record:
                    continue
                
                # Clean up old completed/failed jobs
                if (job_record.status in ["completed", "failed"] and 
                    job_record.updated_at < cutoff_time):
                    
                    if self.delete_job(job_record.job_id):
                        cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old jobs")
            
            return cleaned_count
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self.cleanup_old_jobs()
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("Started background cleanup task")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with self._lock:
            jobs = self.list_jobs()
            stats = {
                "total_jobs": len(jobs),
                "by_status": {},
                "storage_size_mb": 0,
                "oldest_job": None,
                "newest_job": None
            }
            
            # Count by status
            for job in jobs.values():
                stats["by_status"][job.status] = stats["by_status"].get(job.status, 0) + 1
            
            # Calculate storage size
            try:
                total_size = sum(f.stat().st_size for f in self.storage_dir.glob("*.json"))
                stats["storage_size_mb"] = round(total_size / (1024 * 1024), 2)
            except:
                pass
            
            # Find oldest/newest
            if jobs:
                oldest = min(jobs.values(), key=lambda j: j.created_at)
                newest = max(jobs.values(), key=lambda j: j.created_at)
                stats["oldest_job"] = {
                    "id": oldest.job_id,
                    "created": datetime.fromtimestamp(oldest.created_at).isoformat()
                }
                stats["newest_job"] = {
                    "id": newest.job_id,
                    "created": datetime.fromtimestamp(newest.created_at).isoformat()
                }
            
            return stats

# Global instance
job_storage = PersistentJobStorage()
