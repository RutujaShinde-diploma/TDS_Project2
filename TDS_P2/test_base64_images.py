import requests
import json
import time
import base64

def test_api_with_base64_images():
    """Test the API that returns base64 images"""
    
    print("🚀 TESTING API WITH BASE64 IMAGES")
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
                
                # Check if we have structured results with base64 images
                if result.get('result') and hasattr(result['result'], 'charts_data'):
                    print("\n🎯 STRUCTURED ANSWERS WITH BASE64 IMAGES:")
                    print("=" * 60)
                    
                    analysis_result = result['result']
                    
                    # Display answers
                    for i, answer in enumerate(analysis_result.answers, 1):
                        print(f"{i}. Question: {answer.question}")
                        print(f"   Answer: {answer.answer}")
                        if answer.value:
                            print(f"   Value: {answer.value}")
                        if answer.unit:
                            print(f"   Unit: {answer.unit}")
                        print()
                    
                    # Display chart information
                    if analysis_result.charts_generated:
                        print("📈 CHARTS GENERATED:")
                        for chart in analysis_result.charts_generated:
                            print(f"  • {chart}")
                    
                    # Display base64 image data
                    if analysis_result.charts_data:
                        print("\n🖼️ BASE64 IMAGE DATA:")
                        print("=" * 60)
                        for i, chart_data in enumerate(analysis_result.charts_data, 1):
                            print(f"{i}. Filename: {chart_data.filename}")
                            print(f"   Size: {chart_data.size_bytes} bytes")
                            print(f"   Base64 length: {len(chart_data.base64)} characters")
                            print(f"   Data URI: {chart_data.data_uri[:50]}...")
                            
                            # Save base64 image to file for verification
                            try:
                                img_data = base64.b64decode(chart_data.base64)
                                with open(f"downloaded_{chart_data.filename}", 'wb') as f:
                                    f.write(img_data)
                                print(f"   ✅ Saved as: downloaded_{chart_data.filename}")
                            except Exception as e:
                                print(f"   ❌ Failed to save image: {e}")
                            print()
                    
                else:
                    print("\n📋 RAW RESULTS:")
                    print("=" * 60)
                    results = result.get('result', [])
                    for i, result_item in enumerate(results, 1):
                        print(f"{i}. {result_item}")
                
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

def show_expected_response():
    """Show what the improved API response should look like with base64 images"""
    print("\n" + "=" * 60)
    print("🔧 EXPECTED API RESPONSE WITH BASE64 IMAGES:")
    print("=" * 60)
    print("""
{
  "job_id": "e27f9a6e-1200-4d6b-8732-8370cb818760",
  "status": "completed",
  "execution_time": 90.31,
  "result": {
    "answers": [
      {
        "question": "What is the total sales?",
        "answer": "Total sales: 1140",
        "value": "1140",
        "unit": "currency"
      },
      {
        "question": "Which region has the highest sales?",
        "answer": "Region with highest sales: West with total sales: 420",
        "value": "West with total sales: 420"
      }
    ],
    "charts_generated": [
      "bar_chart.png",
      "cumulative_sales.png"
    ],
    "charts_data": [
      {
        "filename": "bar_chart.png",
        "base64": "iVBORw0KGgoAAAANSUhEUgAA...",
        "data_uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        "size_bytes": 12345
      },
      {
        "filename": "cumulative_sales.png",
        "base64": "iVBORw0KGgoAAAANSUhEUgAA...",
        "data_uri": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        "size_bytes": 9876
      }
    ],
    "raw_results": [
      "Total sales: 1140",
      "Region with highest total sales: West with total sales: 420"
    ]
  }
}
""")
    print("=" * 60)

if __name__ == "__main__":
    test_api_with_base64_images()
    show_expected_response()
