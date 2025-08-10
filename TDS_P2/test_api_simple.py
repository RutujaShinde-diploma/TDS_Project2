#!/usr/bin/env python3
"""
Simple test to check OpenAI API connectivity
"""

import os
import asyncio
import openai
from config import config

async def test_openai_api():
    """Test basic OpenAI API connectivity"""
    print("🔑 Testing OpenAI API Connectivity")
    print("=" * 50)
    
    # Check API key
    api_key = config.OPENAI_API_KEY
    if not api_key:
        print("❌ No API key found in config")
        return False
    
    print(f"📋 API key loaded: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else api_key}")
    print(f"🤖 Primary model: {config.OPENAI_MODEL}")
    print(f"🔄 Fallback model: {config.OPENAI_FALLBACK_MODEL}")
    print()
    
    # Test primary model
    print("1️⃣ Testing Primary Model (GPT-4)...")
    try:
        client = openai.AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "user", "content": "What is 2 + 2? Respond with just the number."}
            ],
            max_tokens=10,
            timeout=30
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ Primary model works! Response: '{result}'")
        
        if "4" in result:
            print("✅ Correct answer received")
        else:
            print(f"⚠️ Unexpected answer: {result}")
        
        return True
        
    except openai.APIConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False
    except openai.AuthenticationError as e:
        print(f"❌ Authentication error: {e}")
        return False
    except openai.RateLimitError as e:
        print(f"❌ Rate limit error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

async def test_fallback_model():
    """Test fallback model"""
    print("\n2️⃣ Testing Fallback Model (GPT-3.5-turbo)...")
    
    try:
        client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=config.OPENAI_FALLBACK_MODEL,
            messages=[
                {"role": "user", "content": "What is 3 + 3? Respond with just the number."}
            ],
            max_tokens=10,
            timeout=30
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ Fallback model works! Response: '{result}'")
        
        if "6" in result:
            print("✅ Correct answer received")
        else:
            print(f"⚠️ Unexpected answer: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback model failed: {e}")
        return False

def check_environment():
    """Check environment variables and configuration"""
    print("🔍 Environment Check")
    print("=" * 30)
    
    # Check direct environment variable
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        print(f"✅ OPENAI_API_KEY in environment: {env_key[:10]}...{env_key[-4:] if len(env_key) > 14 else env_key}")
    else:
        print("❌ OPENAI_API_KEY not in environment")
    
    # Check config
    config_key = config.OPENAI_API_KEY
    if config_key:
        print(f"✅ API key in config: {config_key[:10]}...{config_key[-4:] if len(config_key) > 14 else config_key}")
    else:
        print("❌ No API key in config")
    
    # Check if they match
    if env_key and config_key:
        if env_key == config_key:
            print("✅ Environment and config keys match")
        else:
            print("⚠️ Environment and config keys don't match!")
    
    print()

async def main():
    """Run API connectivity tests"""
    print("🔬 OpenAI API Connectivity Test")
    print("=" * 50)
    print()
    
    # Check environment first
    check_environment()
    
    # Test API
    primary_works = await test_openai_api()
    fallback_works = await test_fallback_model()
    
    print("\n📊 Summary")
    print("=" * 20)
    if primary_works:
        print("✅ Primary model (GPT-4) working")
    else:
        print("❌ Primary model failed")
    
    if fallback_works:
        print("✅ Fallback model working")
    else:
        print("❌ Fallback model failed")
    
    if primary_works or fallback_works:
        print("\n🎉 API connectivity confirmed!")
    else:
        print("\n💥 No working API connection found!")

if __name__ == "__main__":
    asyncio.run(main())
