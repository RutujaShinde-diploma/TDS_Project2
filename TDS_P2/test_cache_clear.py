#!/usr/bin/env python3
"""
Test script to demonstrate cache clearing functionality
"""

import requests
import json
import time

def test_cache_management():
    """Test cache management endpoints"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Cache Management Endpoints")
    print("=" * 50)
    
    # 1. Check cache status
    print("\n1. 📊 Checking cache status...")
    try:
        response = requests.get(f"{base_url}/api/cache/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Cache status: {status}")
        else:
            print(f"❌ Failed to get cache status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking cache status: {e}")
    
    # 2. Clear all caches
    print("\n2. 🗑️ Clearing all caches...")
    try:
        response = requests.post(f"{base_url}/api/cache/clear")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Caches cleared: {result}")
        else:
            print(f"❌ Failed to clear caches: {response.status_code}")
    except Exception as e:
        print(f"❌ Error clearing caches: {e}")
    
    # 3. Test with bypass_cache parameter
    print("\n3. 🚀 Testing with cache bypass...")
    
    # Read the test file
    try:
        with open('wiki_test.txt', 'r') as f:
            questions_text = f.read()
        
        print(f"📝 Questions content: {questions_text[:100]}...")
        
        # Submit job with cache bypass
        files = {'questions': ('wiki_test.txt', questions_text, 'text/plain')}
        data = {'bypass_cache': 'true'}
        
        response = requests.post(f'{base_url}/api/', files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Job completed with cache bypass!")
            print(f"📋 Results: {result}")
        else:
            print(f"❌ Failed to submit job: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing cache bypass: {e}")
    
    # 4. Check cache status again
    print("\n4. 📊 Checking cache status after bypass...")
    try:
        response = requests.get(f"{base_url}/api/cache/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Cache status after bypass: {status}")
        else:
            print(f"❌ Failed to get cache status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking cache status: {e}")

def test_specific_plan_cache_clear():
    """Test clearing cache for specific plan content"""
    
    base_url = "http://localhost:8000"
    
    print("\n🔧 Testing Specific Plan Cache Clearing")
    print("=" * 50)
    
    # Read the test file
    try:
        with open('wiki_test.txt', 'r') as f:
            questions_text = f.read()
        
        print(f"📝 Questions content: {questions_text[:100]}...")
        
        # Clear cache for this specific content
        data = {'questions': questions_text}
        response = requests.post(f'{base_url}/api/cache/clear/plan', data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Plan cache cleared: {result}")
        else:
            print(f"❌ Failed to clear plan cache: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error clearing plan cache: {e}")

if __name__ == "__main__":
    print("🚀 Cache Management Test Suite")
    print("Make sure the API server is running: python start_local.py")
    print()
    
    test_cache_management()
    test_specific_plan_cache_clear()
    
    print("\n✅ Cache management tests completed!")
