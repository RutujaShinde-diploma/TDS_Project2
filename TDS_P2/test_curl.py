import requests
import json

def test_wiki_curl():
    """Test Wikipedia scraping with curl-like approach"""
    
    print("🌐 TESTING WIKIPEDIA SCRAPING (CURL-STYLE)")
    print("=" * 50)
    
    # API endpoint
    url = "http://localhost:8000/api/"
    
    # Prepare files for upload (like curl -F)
    files = [
        ('questions', ('wiki_test.txt', open('wiki_test.txt', 'rb'), 'text/plain')),
        ('files', ('', '', 'text/plain'))  # Empty file for scraping
    ]
    
    try:
        # Submit job (equivalent to curl -X POST)
        print("📤 Submitting job...")
        print(f"URL: {url}")
        print(f"Files: questions=@wiki_test.txt, files=")
        
        response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id')
            print(f"✅ Job ID: {job_id}")
            
            # Get job status (equivalent to curl GET)
            status_url = f"http://localhost:8000/api/job/{job_id}"
            print(f"\n📊 Checking status: {status_url}")
            
            status_response = requests.get(status_url)
            print(f"Status Code: {status_response.status_code}")
            print(f"Status Response: {status_response.text}")
            
        else:
            print("❌ Job submission failed")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_wiki_curl()
