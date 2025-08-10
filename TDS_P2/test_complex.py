import requests
import json
import time

def test_complex_analysis():
    """Test the API with complex data analysis"""
    
    # Read the test file
    with open('complex_test.txt', 'r') as f:
        questions_text = f.read()
    
    print("📊 Testing API with complex data analysis...")
    print(f"Questions: {questions_text[:200]}...")
    print("-" * 60)
    
    # Submit job
    files = {'questions': ('complex_test.txt', questions_text, 'text/plain')}
    
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
        print("-" * 60)
        
        # Poll for results
        max_attempts = 30
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
                        expected = ["4", "62125", "Engineering"]
                        for i, (exp, act) in enumerate(zip(expected, results), 1):
                            status = "✅" if str(act) == exp else "❌"
                            print(f"  {status} Q{i}: Expected {exp}, Got {act}")
                        
                        print("\n📈 Analysis:")
                        print(f"  • Engineering employees: {results[0] if len(results) > 0 else 'N/A'}")
                        print(f"  • Average salary: ${results[1] if len(results) > 1 else 'N/A'}")
                        print(f"  • Highest avg salary dept: {results[2] if len(results) > 2 else 'N/A'}")
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
    test_complex_analysis()
