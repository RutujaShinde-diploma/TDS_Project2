import requests
import json
import time

def test_basic_math():
    """Test the API with basic math calculations"""
    
    # Read the test file
    with open('basic_math_test.txt', 'r') as f:
        questions_text = f.read()
    
    print("🧮 Testing API with basic math...")
    print(f"Questions: {questions_text}")
    print("-" * 50)
    
    # Submit job
    files = {'questions': ('basic_math_test.txt', questions_text, 'text/plain')}
    
    try:
        response = requests.post('http://localhost:8000/api/', files=files, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Failed to submit job: {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        result = response.json()
        job_id = result['job_id']
        print(f"✅ Job submitted successfully!")
        print(f"📋 Job ID: {job_id}")
        print("-" * 50)
        
        # Poll for results
        max_attempts = 20
        for attempt in range(max_attempts):
            time.sleep(3)  # Wait 3 seconds between checks
            
            try:
                response = requests.get(f'http://localhost:8000/api/job/{job_id}', timeout=10)
                
                if response.status_code != 200:
                    print(f"❌ Failed to get job status: {response.status_code}")
                    continue
                
                result = response.json()
                status = result['status']
                
                print(f"📊 Attempt {attempt + 1}: Status = {status}")
                
                if status == 'completed':
                    print("🎉 Job completed successfully!")
                    print(f"⏱️ Execution time: {result.get('execution_time', 'N/A')}")
                    print(f"📋 Results: {result.get('result', 'No results')}")
                    
                    # Validate results
                    results = result.get('result', [])
                    if results:
                        print("\n✅ Expected vs Actual:")
                        expected = ["4", "15", "6"]
                        for i, (exp, act) in enumerate(zip(expected, results), 1):
                            status = "✅" if str(act) == exp else "❌"
                            print(f"  {status} Q{i}: Expected {exp}, Got {act}")
                    return
                    
                elif status == 'failed':
                    print(f"❌ Job failed: {result.get('error', 'Unknown error')}")
                    return
                    
                elif status == 'processing':
                    print("⏳ Still processing...")
                    
                else:
                    print(f"❓ Unknown status: {status}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Network error: {e}")
                continue
        
        print("⏰ Timeout - job took too long")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to server: {e}")
        print("💡 Make sure the server is running with: python start_local.py")

if __name__ == "__main__":
    test_basic_math()
