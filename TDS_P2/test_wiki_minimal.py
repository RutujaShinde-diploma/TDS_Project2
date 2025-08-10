import requests
import json
import time

def test_wiki_scraping():
    """Minimal test for Wikipedia scraping"""
    
    print("🌐 TESTING WIKIPEDIA SCRAPING")
    print("=" * 50)
    
    # API endpoint
    url = "http://localhost:8000/api/"
    
    # Prepare files for upload
    files = {
        'questions': ('wiki_test.txt', open('wiki_test.txt', 'rb'), 'text/plain'),
        'files': ('', '', 'text/plain')  # No file upload needed for scraping
    }
    
    try:
        # Submit job
        print("📤 Submitting Wikipedia scraping job...")
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
        print("-" * 50)
        
        # Poll for results (shorter timeout for minimal test)
        max_attempts = 10
        for attempt in range(max_attempts):
            time.sleep(3)
            
            response = requests.get(f'http://localhost:8000/api/job/{job_id}')
            if response.status_code != 200:
                print(f"❌ Failed to get job status: {response.status_code}")
                return
            
            result = response.json()
            status = result['status']
            
            print(f"📊 Attempt {attempt + 1}: Status = {status}")
            
            if status == "completed":
                print("🎉 Wikipedia scraping completed!")
                print(f"⏱️ Execution time: {result.get('execution_time', 'N/A')}")
                
                # Show results
                if result.get('result'):
                    print("\n📋 RESULTS:")
                    print("=" * 50)
                    if isinstance(result['result'], dict):
                        for key, value in result['result'].items():
                            if key.startswith('answer'):
                                if 'base64' in str(value).lower() or len(str(value)) > 100:
                                    print(f"{key}: [BASE64_IMAGE - {len(str(value))} chars]")
                                else:
                                    print(f"{key}: {value}")
                    else:
                        print(f"Result: {result['result']}")
                
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

if __name__ == "__main__":
    test_wiki_scraping()
