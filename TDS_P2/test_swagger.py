#!/usr/bin/env python3
"""
Test script for Swagger UI testing
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_api_with_file():
    """Test the API with the wiki_test.txt file"""
    
    print("🧪 Testing API with wiki_test.txt file")
    print("=" * 50)
    
    # Check API health
    print("1. Checking API health...")
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API is healthy")
        else:
            print(f"❌ API health check failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False
    
    # Submit request with file
    print("\n2. Submitting request with wiki_test.txt...")
    
    try:
        with open('wiki_test.txt', 'rb') as f:
            files = {'questions': ('wiki_test.txt', f, 'text/plain')}
            response = requests.post(f"{API_URL}/api/", files=files, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        result = response.json()
        job_id = result['job_id']
        print(f"✅ Job submitted successfully")
        print(f"📋 Job ID: {job_id}")
        
        # Poll for results
        print(f"\n3. Processing analysis (max 2 minutes)...")
        max_wait_time = 120  # 2 minutes
        poll_interval = 5
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                status_response = requests.get(f"{API_URL}/api/job/{job_id}", timeout=30)
                
                if status_response.status_code != 200:
                    print(f"❌ Error checking status: {status_response.status_code}")
                    break
                
                status_result = status_response.json()
                current_status = status_result['status']
                elapsed = time.time() - start_time
                
                print(f"⏱️  [{elapsed:.0f}s] Status: {current_status}")
                
                if current_status == 'completed':
                    print(f"\n4. ✅ Analysis completed in {elapsed:.1f}s!")
                    
                    results = status_result.get('result', [])
                    print(f"\n📊 Results ({len(results)} items):")
                    
                    for i, result_item in enumerate(results, 1):
                        print(f"  {i}. {result_item}")
                    
                    return True
                
                elif current_status == 'failed':
                    print(f"\n❌ Analysis failed!")
                    error = status_result.get('error', 'Unknown error')
                    print(f"Error: {error}")
                    return False
                
                time.sleep(poll_interval)
                
            except Exception as e:
                print(f"❌ Error polling status: {e}")
                break
        
        print(f"\n❌ Test timed out after {max_wait_time}s")
        return False
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_api_with_file()
    if success:
        print("\n🎉 Test PASSED!")
    else:
        print("\n💥 Test FAILED!")
