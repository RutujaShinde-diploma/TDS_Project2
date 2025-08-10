import requests
import json
import time

def get_job_results(job_id="09a18b67-32c6-406f-9386-14782029f0ab"):
    """Get results for a specific job"""
    
    print(f"📊 Checking results for job: {job_id}")
    print("=" * 60)
    
    # Poll for results
    max_attempts = 30  # 60 seconds total
    for attempt in range(max_attempts):
        try:
            response = requests.get(f'http://localhost:8000/api/job/{job_id}')
            
            if response.status_code == 200:
                result = response.json()
                status = result['status']
                
                print(f"📊 Attempt {attempt + 1}: Status = {status}")
                
                if status == "completed":
                    print("🎉 Job completed successfully!")
                    print(f"⏱️ Execution time: {result.get('execution_time', 'N/A')}")
                    
                    # Display the actual results
                    results = result.get('result', 'No results')
                    print("\n" + "=" * 60)
                    print("📋 ANALYSIS RESULTS:")
                    print("=" * 60)
                    
                    if isinstance(results, list):
                        print("📊 Sales Analysis Results:")
                        for i, result_item in enumerate(results, 1):
                            print(f"  {i}. {result_item}")
                    elif isinstance(results, dict):
                        print("📊 Sales Analysis Results:")
                        for key, value in results.items():
                            print(f"  • {key}: {value}")
                    else:
                        print(f"📊 Result: {results}")
                    
                    return results
                    
                elif status == "error":
                    print(f"❌ Job failed: {result.get('error', 'Unknown error')}")
                    return None
                elif status in ["processing", "executing", "planning_complete"]:
                    print("⏳ Still processing...")
                else:
                    print(f"❓ Unknown status: {status}")
            else:
                print(f"❌ Failed to get job status: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
            
        time.sleep(2)
    
    print("⏰ Timeout: Job did not complete within expected time")
    return None

def show_expected_results():
    """Show what results we expect from the analysis"""
    print("\n" + "=" * 60)
    print("📋 EXPECTED RESULTS FROM SALES ANALYSIS:")
    print("=" * 60)
    print("Based on questions_csv.txt, we should get:")
    print("1. total_sales: Total sales across all regions")
    print("2. top_region: Region with highest total sales")
    print("3. day_sales_correlation: Correlation between day of month and sales")
    print("4. bar_chart: Base64 PNG of total sales by region")
    print("5. median_sales: Median sales amount across all orders")
    print("6. total_sales_tax: Total sales tax (10% rate)")
    print("7. cumulative_sales_chart: Base64 PNG of cumulative sales over time")
    print("=" * 60)

if __name__ == "__main__":
    show_expected_results()
    results = get_job_results()
    
    if results:
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ Could not retrieve results")
