#!/usr/bin/env python3
"""
Render startup script for the Data Analyst Agent API
This script handles environment setup and starts the FastAPI server
"""

import os
import sys
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment for Render deployment"""
    logger.info("Setting up environment for Render deployment...")
    
    # Create necessary directories
    workspace_dir = Path("workspace")
    logs_dir = Path("logs")
    
    workspace_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created workspace directory: {workspace_dir.absolute()}")
    logger.info(f"Created logs directory: {logs_dir.absolute()}")
    
    # Set environment variables if not already set
    if not os.getenv("WORKSPACE_DIR"):
        os.environ["WORKSPACE_DIR"] = str(workspace_dir.absolute())
        logger.info(f"Set WORKSPACE_DIR to: {workspace_dir.absolute()}")
    
    if not os.getenv("LOG_LEVEL"):
        os.environ["LOG_LEVEL"] = "INFO"
        logger.info("Set LOG_LEVEL to: INFO")
    
    # Check required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning("These must be set in Render dashboard")
    else:
        logger.info("All required environment variables are set")
    
    logger.info("Environment setup completed")

def main():
    """Main startup function"""
    try:
        setup_environment()
        
        # Import and start the FastAPI app
        from main import app
        import uvicorn
        
        port = int(os.getenv("PORT", "8000"))
        logger.info(f"Starting FastAPI server on port {port}")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
