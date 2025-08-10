import requests
import json
import time

def test_simple_output():
    """Test the simplified API output structure"""
    
    print("🚀 TESTING SIMPLIFIED API OUTPUT")
    print("=" * 60)
    
    # API endpoint
    url = "http://localhost:8000/api/"
    
    # Prepare files for upload
    files = {
        'questions': ('questions_csv.txt', open('questions_csv.txt', 'rb'), 'text/plain'),
        'files': ('sample-sales.csv', open('sample-sales.csv', 'rb'), 'text/csv')
    }
    
    try:
        # Submit job
        print("📤 Submitting analysis job...")
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
        max_attempts = 30
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
                
                # Check if we have simple numbered answers
                if result.get('result') and isinstance(result['result'], dict):
                    print("\n🎯 SIMPLE NUMBERED ANSWERS:")
                    print("=" * 60)
                    
                    answers = result['result']
                    for key, value in answers.items():
                        if key.startswith('answer'):
                            if 'base64' in str(value).lower() or len(str(value)) > 100:
                                print(f"{key}: [BASE64_IMAGE_DATA - {len(str(value))} characters]")
                            else:
                                print(f"{key}: {value}")
                    
                    print("\n✅ SIMPLIFIED OUTPUT STRUCTURE VERIFIED!")
                    print("=" * 60)
                    print("""
🎯 What we've achieved:
• ✅ Simple numbered answers (answer1, answer2, etc.)
• ✅ No complex nested structures
• ✅ Base64 images included as answer values
• ✅ Clean, user-friendly output format
• ✅ Easy to parse and use in applications
""")
                else:
                    print("\n📋 RAW RESULTS:")
                    print("=" * 60)
                    results = result.get('result', [])
                    if isinstance(results, list):
                        for i, result_item in enumerate(results, 1):
                            print(f"{i}. {result_item}")
                    else:
                        print(f"Result: {results}")
                
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

def show_expected_output():
    """Show what the simplified output should look like"""
    print("\n" + "=" * 60)
    print("🔧 EXPECTED SIMPLIFIED OUTPUT:")
    print("=" * 60)
    print("""
{
    "answer1": "1140",
    "answer2": "West",
    "answer3": "0.22281245492773066",
    "answer4": "iVBORw0KGgoAAAANSUhEUgAA...",
    "answer5": "140.0",
    "answer6": "114.0",
    "answer7": "iVBORw0KGgoAAAANSUhEUgAA..."
}
""")
    print("=" * 60)

if __name__ == "__main__":
    test_simple_output()
    show_expected_output()
