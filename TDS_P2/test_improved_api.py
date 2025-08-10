import requests
import json
import time

def test_improved_api():
    """Test the improved API that returns structured answers"""
    
    print("🚀 TESTING IMPROVED API WITH STRUCTURED ANSWERS")
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
                
                # Check if we have structured results
                if result.get('result') and hasattr(result['result'], 'answers'):
                    print("\n🎯 STRUCTURED ANSWERS:")
                    print("=" * 60)
                    
                    analysis_result = result['result']
                    for i, answer in enumerate(analysis_result.answers, 1):
                        print(f"{i}. Question: {answer.question}")
                        print(f"   Answer: {answer.answer}")
                        if answer.value:
                            print(f"   Value: {answer.value}")
                        if answer.unit:
                            print(f"   Unit: {answer.unit}")
                        print()
                    
                    if analysis_result.charts_generated:
                        print("📈 CHARTS GENERATED:")
                        for chart in analysis_result.charts_generated:
                            print(f"  • {chart}")
                    
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
    """Show what the improved API response should look like"""
    print("\n" + "=" * 60)
    print("🔧 EXPECTED IMPROVED API RESPONSE:")
    print("=" * 60)
    print("""
{
  "job_id": "09a18b67-32c6-406f-9386-14782029f0ab",
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
      },
      {
        "question": "What is the correlation between day of month and sales?",
        "answer": "Correlation: 0.22281245492773066",
        "value": "0.22281245492773066"
      },
      {
        "question": "What is the median sales?",
        "answer": "Median sales: 140.0",
        "value": "140.0",
        "unit": "currency"
      },
      {
        "question": "What is the total sales tax?",
        "answer": "Total sales tax: 114.0",
        "value": "114.0",
        "unit": "currency"
      }
    ],
    "charts_generated": [
      "sales_by_region.png",
      "cumulative_sales.png"
    ],
    "raw_results": [
      "Total sales: 1140",
      "Region with highest total sales: West with total sales: 420",
      "Correlation between day of month and sales: 0.22281245492773066",
      "Median sales: 140.0",
      "Total sales tax: 114.0"
    ]
  }
}
""")
    print("=" * 60)

if __name__ == "__main__":
    test_improved_api()
    show_expected_response()
