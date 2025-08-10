import logging
import sys
from pathlib import Path
from pythonjsonlogger import jsonlogger

from config import config

def setup_logger(name: str) -> logging.Logger:
    """Setup structured logging"""
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    
    # Create formatter
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
