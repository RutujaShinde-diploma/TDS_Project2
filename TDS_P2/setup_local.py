#!/usr/bin/env python3
"""
Setup script for local development environment
"""

import os
import sys
import subprocess
import platform

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✅ {env_file} already exists")
        return True
    
    print(f"📝 Creating {env_file} file...")
    
    api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        print("❌ OpenAI API key is required")
        return False
    
    env_content = f"""# Data Analyst Agent API Configuration
OPENAI_API_KEY={api_key}
OPENAI_MODEL=gpt-4
OPENAI_FALLBACK_MODEL=gpt-3.5-turbo
REDIS_URL=memory://localhost
MAX_EXECUTION_TIME=180
MAX_FILE_SIZE=100000000
SANDBOX_TIMEOUT=30
LOG_LEVEL=INFO
CACHE_TTL=3600
USE_DOCKER=false
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Created {env_file}")
    return True

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    directories = ["workspace", "logs"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def load_env_variables():
    """Load environment variables from .env file"""
    env_file = ".env"
    if not os.path.exists(env_file):
        return
    
    print("🔧 Loading environment variables...")
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
    
    print("✅ Environment variables loaded")

def main():
    """Main setup function"""
    print("🚀 Setting up Data Analyst Agent API for Local Development")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create directories
    setup_directories()
    
    # Create .env file
    if not create_env_file():
        sys.exit(1)
    
    # Load environment variables
    load_env_variables()
    
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("\n🎯 Next steps:")
    print("1. Run the server: python start_local.py")
    print("2. Test the API: python test_example.py")
    print("3. Visit API docs: http://localhost:8000/docs")
    print("\n📝 Notes:")
    print("- Running without Docker (subprocess execution)")
    print("- Using in-memory cache (no Redis required)")
    print("- All data stored in ./workspace directory")

if __name__ == "__main__":
    main()
