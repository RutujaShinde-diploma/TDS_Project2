import requests
import json
import time

def test_csv_with_swagger_format():
    """Test CSV analysis using the same format as Swagger UI"""
    
    print("📊 Testing CSV Analysis (Swagger UI Format)...")
    print("=" * 60)
    
    # API endpoint
    url = "http://localhost:8000/api/"
    
    # Prepare multipart form data (same as Swagger UI)
    files = {
        'questions': ('questions_csv.txt', open('questions_csv.txt', 'rb'), 'text/plain'),
        'files': ('sample-sales.csv', open('sample-sales.csv', 'rb'), 'text/csv')
    }
    
    # Headers (same as Swagger UI)
    headers = {
        'accept': 'application/json',
        'Content-Type': 'multipart/form-data'
    }
    
    try:
        print("📤 Submitting request...")
        print(f"📁 Files being uploaded:")
        print(f"  • questions: questions_csv.txt")
        print(f"  • files: sample-sales.csv")
        print("-" * 60)
        
        # Submit request
        response = requests.post(url, files=files, headers=headers)
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            job_id = result['job_id']
            status = result['status']
            
            print(f"✅ Success! Job ID: {job_id}")
            print(f"📊 Initial Status: {status}")
            print("-" * 60)
            
            # Poll for completion
            print("⏳ Polling for results...")
            for attempt in range(15):  # 30 seconds
                time.sleep(2)
                
                status_response = requests.get(f'http://localhost:8000/api/job/{job_id}')
                if status_response.status_code == 200:
                    job_result = status_response.json()
                    current_status = job_result['status']
                    
                    print(f"📊 Attempt {attempt + 1}: {current_status}")
                    
                    if current_status == "completed":
                        print("🎉 Analysis completed!")
                        print(f"⏱️ Execution time: {job_result.get('execution_time', 'N/A')}")
                        
                        results = job_result.get('result', 'No results')
                        print(f"📋 Results: {results}")
                        
                        # Parse and display results nicely
                        if isinstance(results, list):
                            print("\n📊 Analysis Results:")
                            for i, result in enumerate(results, 1):
                                print(f"  {i}. {result}")
                        else:
                            print(f"\n📊 Single Result: {results}")
                        
                        return
                    elif current_status == "error":
                        print(f"❌ Error: {job_result.get('error', 'Unknown error')}")
                        return
                else:
                    print(f"❌ Failed to get status: {status_response.status_code}")
                    return
            
            print("⏰ Timeout: Job did not complete in time")
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Make sure the server is running: python start_local.py")

def show_curl_command():
    """Show the equivalent curl command for Swagger UI"""
    print("\n" + "=" * 60)
    print("🔧 Equivalent cURL Command for Swagger UI:")
    print("=" * 60)
    print("""
curl -X 'POST' \\
  'http://localhost:8000/api/' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: multipart/form-data' \\
  -F 'questions=@questions_csv.txt;type=text/plain' \\
  -F 'files=@sample-sales.csv;type=text/csv'
""")
    print("=" * 60)

if __name__ == "__main__":
    test_csv_with_swagger_format()
    show_curl_command()
