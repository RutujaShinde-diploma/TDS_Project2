#!/usr/bin/env python3
"""
Simple test to check API functionality
"""

import requests
import json
import time

def test_simple_api():
    """Test the API with a simple dataset"""
    
    # Read the test file
    with open('simple_test.txt', 'r') as f:
        questions_text = f.read()
    
    print("🚀 Testing API with simple dataset...")
    print(f"Questions: {questions_text[:100]}...")
    
    # Submit job
    files = {'questions': ('simple_test.txt', questions_text, 'text/plain')}
    response = requests.post('http://localhost:8000/api/', files=files)
    
    if response.status_code != 200:
        print(f"❌ Failed to submit job: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    job_id = result['job_id']
    print(f"✅ Job submitted: {job_id}")
    
    # Poll for results
    max_attempts = 30
    for attempt in range(max_attempts):
        time.sleep(2)
        
        response = requests.get(f'http://localhost:8000/api/job/{job_id}')
        if response.status_code != 200:
            print(f"❌ Failed to get job status: {response.status_code}")
            continue
        
        result = response.json()
        status = result['status']
        
        print(f"📊 Attempt {attempt + 1}: Status = {status}")
        
        if status == 'completed':
            print("✅ Job completed!")
            print(f"📈 Execution time: {result.get('execution_time', 'N/A')}")
            print(f"📋 Results: {result.get('result', 'No results')}")
            return
        elif status == 'failed':
            print(f"❌ Job failed: {result.get('error', 'Unknown error')}")
            return
        elif status == 'processing':
            print("⏳ Still processing...")
        else:
            print(f"❓ Unknown status: {status}")
    
    print("⏰ Timeout - job took too long")

if __name__ == "__main__":
    test_simple_api()
