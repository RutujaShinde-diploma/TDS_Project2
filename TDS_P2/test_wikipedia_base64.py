import requests
import json
import time
import base64

def test_wikipedia_with_base64_images():
    """Test the Wikipedia example with base64 images"""
    
    print("🚀 TESTING WIKIPEDIA EXAMPLE WITH BASE64 IMAGES")
    print("=" * 60)
    
    # API endpoint
    url = "http://localhost:8000/api/"
    
    # Prepare files for upload (Wikipedia test)
    files = {
        'questions': ('wiki_test.txt', open('wiki_test.txt', 'rb'), 'text/plain'),
        'files': ('', '', 'text/plain')  # Empty file for scraping
    }
    
    try:
        # Submit job
        print("📤 Submitting Wikipedia analysis job...")
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
        max_attempts = 60  # More attempts for Wikipedia scraping
        for attempt in range(max_attempts):
            time.sleep(3)  # Longer wait for scraping
            
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
                
                # Check if we have structured results with base64 images
                if result.get('result') and isinstance(result['result'], dict) and 'charts_data' in result['result']:
                    print("\n🎯 STRUCTURED ANSWERS WITH BASE64 IMAGES:")
                    print("=" * 60)
                    
                    analysis_result = result['result']
                    
                    # Display answers
                    for i, answer in enumerate(analysis_result.get('answers', []), 1):
                        print(f"{i}. Question: {answer.get('question', 'N/A')}")
                        print(f"   Answer: {answer.get('answer', 'N/A')}")
                        if answer.get('value'):
                            print(f"   Value: {answer.get('value')}")
                        if answer.get('unit'):
                            print(f"   Unit: {answer.get('unit')}")
                        print()
                    
                    # Display chart information
                    if analysis_result.get('charts_generated'):
                        print("📈 CHARTS GENERATED:")
                        for chart in analysis_result['charts_generated']:
                            print(f"  • {chart}")
                    
                    # Display base64 image data
                    if analysis_result.get('charts_data'):
                        print("\n🖼️ BASE64 IMAGE DATA:")
                        print("=" * 60)
                        for i, chart_data in enumerate(analysis_result['charts_data'], 1):
                            print(f"{i}. Filename: {chart_data.get('filename', 'N/A')}")
                            print(f"   Size: {chart_data.get('size_bytes', 'N/A')} bytes")
                            base64_str = chart_data.get('base64', '')
                            print(f"   Base64 length: {len(base64_str)} characters")
                            data_uri = chart_data.get('data_uri', '')
                            print(f"   Data URI: {data_uri[:50]}...")
                            
                            # Save base64 image to file for verification
                            try:
                                img_data = base64.b64decode(base64_str)
                                filename = chart_data.get('filename', f'chart_{i}.png')
                                with open(f"downloaded_{filename}", 'wb') as f:
                                    f.write(img_data)
                                print(f"   ✅ Saved as: downloaded_{filename}")
                            except Exception as e:
                                print(f"   ❌ Failed to save image: {e}")
                            print()
                    
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

def show_wikipedia_expected_response():
    """Show what the Wikipedia API response should look like with base64 images"""
    print("\n" + "=" * 60)
    print("🔧 EXPECTED WIKIPEDIA API RESPONSE WITH BASE64 IMAGES:")
    print("=" * 60)
    print("""
{
  "job_id": "abc123-def456-ghi789",
  "status": "completed",
  "execution_time": 120.45,
  "result": {
    "answers": [
      {
        "question": "How many movies grossed over $2 billion before 2000?",
        "answer": "Count: 0 movies",
        "value": "0",
        "unit": "movies"
      },
      {
        "question": "What is the earliest movie to gross over $1.5 billion?",
        "answer": "Earliest movie: Titanic (1997)",
        "value": "Titanic (1997)"
      }
    ],
    "charts_generated": [
      "highest_grossing_movies.png",
      "revenue_timeline.png"
    ],
    "charts_data": [
      {
        "filename": "highest_grossing_movies.png",
        "base64": "iVBORw0KGgoAAAANSUhEUgAA...",
        "data_uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        "size_bytes": 15678
      },
      {
        "filename": "revenue_timeline.png",
        "base64": "iVBORw0KGgoAAAANSUhEUgAA...",
        "data_uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        "size_bytes": 12345
      }
    ],
    "raw_results": [
      "Count: 0 movies grossed over $2 billion before 2000",
      "Earliest movie to gross over $1.5 billion: Titanic (1997)"
    ]
  }
}
""")
    print("=" * 60)

if __name__ == "__main__":
    test_wikipedia_with_base64_images()
    show_wikipedia_expected_response()
