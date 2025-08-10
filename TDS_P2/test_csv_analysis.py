import requests
import json
import time

def test_csv_analysis():
    """Test the API with CSV file upload and analysis"""
    
    print("📊 Testing API with CSV file analysis...")
    
    # API endpoint
    url = "http://localhost:8000/api/"
    
    # Prepare files for upload
    files = {
        'questions': ('questions_csv.txt', open('questions_csv.txt', 'rb'), 'text/plain'),
        'files': ('sample-sales.csv', open('sample-sales.csv', 'rb'), 'text/csv')
    }
    
    try:
        # Submit job
        print("📤 Submitting job with CSV file...")
        response = requests.post(url, files=files)
        
        if response.status_code != 200:
            print(f"❌ Failed to submit job: {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        # Parse response
        result = response.json()
        job_id = result['job_id']
        
        print("✅ Job submitted successfully!")
        print(f"📋 Job ID: {job_id}")
        print("-" * 60)
        
        # Poll for results
        max_attempts = 30  # 60 seconds total
        for attempt in range(max_attempts):
            time.sleep(2)
            
            response = requests.get(f'http://localhost:8000/api/job/{job_id}')
            if response.status_code != 200:
                print(f"❌ Failed to get job status: {response.status_code}")
                return
            
            result = response.json()
            status = result['status']
            
            print(f"📊 Attempt {attempt + 1}: Status = {status}")
            
            if status == "completed":
                print("🎉 Job completed successfully!")
                print(f"⏱️ Execution time: {result.get('execution_time', 'N/A')}")
                print(f"📋 Results: {result.get('result', 'No results')}")
                
                # Validate results
                if result.get('result'):
                    results = result['result']
                    print("\n✅ Results Analysis:")
                    print(f"  • Total results returned: {len(results) if isinstance(results, list) else 'Single result'}")
                    print(f"  • Result type: {type(results)}")
                    print(f"  • Result content: {results}")
                else:
                    print("❌ No results returned")
                
                return
            elif status == "error":
                print(f"❌ Job failed: {result.get('error', 'Unknown error')}")
                return
            elif status in ["processing", "executing"]:
                print("⏳ Still processing...")
            else:
                print(f"❓ Unknown status: {status}")
        
        print("⏰ Timeout: Job did not complete within expected time")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Make sure the server is running with: python start_local.py")

if __name__ == "__main__":
    test_csv_analysis()
