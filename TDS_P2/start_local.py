#!/usr/bin/env python3
"""
Local development server for Data Analyst Agent API
Runs without Docker for testing purposes
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import openai
        import pandas
        import matplotlib
        import requests
        from bs4 import BeautifulSoup  # beautifulsoup4 is imported as bs4
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def setup_environment():
    """Setup local environment"""
    # Create necessary directories
    os.makedirs("workspace", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Set environment variables for local development
    os.environ["PYTHONPATH"] = str(Path(__file__).parent)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    
    # Use simple in-memory cache (no Redis required)
    os.environ["CACHE_TYPE"] = "simple"
    
    print("✅ Environment setup complete")

def load_api_key():
    """Load API key from file"""
    key_files = ["api_key.txt", "openai_key.txt"]
    
    for key_file in key_files:
        if os.path.exists(key_file):
            try:
                with open(key_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                    if api_key:
                        os.environ["OPENAI_API_KEY"] = api_key
                        print(f"✅ OpenAI API key loaded from {key_file}")
                        return True
            except Exception as e:
                print(f"⚠️  Could not read {key_file}: {e}")
    
    # Check if already in environment
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OpenAI API key found in environment")
        return True
    
    print("❌ OPENAI_API_KEY not found")
    print("Please create api_key.txt with your OpenAI API key")
    return False

def start_simple_cache():
    """Setup simple in-memory cache (no Redis required)"""
    print("✅ Using simple in-memory cache (no Redis required)")
    os.environ["CACHE_TYPE"] = "simple"
    return True

def load_env_variables():
    """Load environment variables from .env file"""
    env_file = ".env"
    if not os.path.exists(env_file):
        return
    
    print("🔧 Loading environment variables...")
    try:
        # Try using python-dotenv if available
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print("✅ Environment variables loaded via python-dotenv")
    except ImportError:
        # Fallback to manual parsing
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
        print("✅ Environment variables loaded manually")

def main():
    """Main function to start the local server"""
    print("🚀 Starting Data Analyst Agent API (Local Mode)")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Load API key from file
    if not load_api_key():
        sys.exit(1)
    
    # Setup simple cache (no Redis required)
    start_simple_cache()
    
    print("\n📡 Starting API server...")
    print("API will be available at: http://localhost:8000")
    print("Health check: http://localhost:8000/health")
    print("API docs: http://localhost:8000/docs")
    print("🔒 Auto-reload disabled for stable job execution")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Import and run the app (auto-reload disabled to prevent job data loss)
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
    
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
