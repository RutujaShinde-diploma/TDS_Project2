#!/usr/bin/env python3
"""
Test script to verify improved Wikipedia data handling
"""

import requests
import json
import time

def test_improved_wiki_analysis():
    """Test the improved Wikipedia analysis with better data cleaning"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Improved Wikipedia Analysis")
    print("=" * 50)
    
    # Test with cache bypass to ensure fresh execution
    print("\n1. 🚀 Testing with improved data cleaning...")
    
    try:
        # Read the test file
        with open('wiki_test.txt', 'r') as f:
            questions_text = f.read()
        
        print(f"📝 Questions: {questions_text[:100]}...")
        
        # Submit job with cache bypass
        files = {'questions': ('wiki_test.txt', questions_text, 'text/plain')}
        data = {'bypass_cache': 'true'}
        
        print("📤 Submitting job with cache bypass...")
        response = requests.post(f'{base_url}/api/', files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Job completed successfully!")
            print(f"📋 Results: {json.dumps(result, indent=2)}")
            
            # Analyze results
            print("\n🔍 Result Analysis:")
            for key, value in result.items():
                if "error" in str(value).lower():
                    print(f"❌ {key}: {value}")
                else:
                    print(f"✅ {key}: {value}")
                    
        else:
            print(f"❌ Failed to submit job: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing improved analysis: {e}")

def test_cache_status():
    """Check cache status after the test"""
    
    base_url = "http://localhost:8000"
    
    print("\n2. 📊 Checking cache status...")
    
    try:
        response = requests.get(f"{base_url}/api/cache/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Cache status: {status}")
        else:
            print(f"❌ Failed to get cache status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking cache status: {e}")

if __name__ == "__main__":
    print("🚀 Improved Wikipedia Analysis Test")
    print("Make sure the API server is running: python start_local.py")
    print()
    
    test_improved_wiki_analysis()
    test_cache_status()
    
    print("\n✅ Test completed!")
    print("\n💡 If you still see errors, the improvements should help with:")
    print("   - Better data cleaning for Wikipedia tables")
    print("   - Proper handling of mixed data types")
    print("   - Removal of footnotes and special characters")
    print("   - Data validation before calculations")
