import os
from typing import Optional

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    def __init__(self):
        # Auto-load API key if not already in environment
        if not self.OPENAI_API_KEY:
            self._load_api_key()
    
    def _load_api_key(self):
        """Load API key from file if not in environment"""
        key_files = ["api_key.txt", "openai_key.txt"]
        
        for key_file in key_files:
            if os.path.exists(key_file):
                try:
                    with open(key_file, 'r', encoding='utf-8') as f:
                        api_key = f.read().strip()
                        if api_key:
                            os.environ["OPENAI_API_KEY"] = api_key
                            self.OPENAI_API_KEY = api_key
                            return
                except Exception:
                    continue
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_FALLBACK_MODEL: str = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-3.5-turbo")
    
    # API Configuration
    MAX_EXECUTION_TIME: int = int(os.getenv("MAX_EXECUTION_TIME", "180"))  # 3 minutes
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "100000000"))  # 100MB
    MAX_RETRIES: int = 3
    
    # Sandbox Configuration
    SANDBOX_TIMEOUT: int = int(os.getenv("SANDBOX_TIMEOUT", "30"))
    
    # Cache Configuration
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # File Configuration - Use relative path for production compatibility
    WORKSPACE_DIR: str = os.getenv("WORKSPACE_DIR", "workspace")
    ALLOWED_EXTENSIONS = {
        ".txt", ".csv", ".json", ".parquet", ".xlsx", ".xls", 
        ".png", ".jpg", ".jpeg", ".pdf", ".html", ".xml"
    }
    
    # Safety Configuration (reduced restrictions for legitimate scraping)
    BLOCKED_IMPORTS = {
        "subprocess", "importlib", "exec", "eval",
        "__import__", "compile", "globals", "locals"
    }
    
    ALLOWED_DOMAINS = {
        "wikipedia.org", "en.wikipedia.org", "api.github.com",
        "httpbin.org", "jsonplaceholder.typicode.com"
    }

config = Config()
