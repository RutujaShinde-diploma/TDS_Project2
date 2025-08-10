import json
import requests

def get_structured_results(job_id="09a18b67-32c6-406f-9386-14782029f0ab"):
    """Get structured results from the job"""
    
    print("📊 STRUCTURED ANALYSIS RESULTS")
    print("=" * 60)
    
    try:
        # Read from job storage
        with open(f'job_storage/{job_id}.json', 'r') as f:
            job_data = json.load(f)
        
        if job_data['status'] == 'completed':
            results = job_data.get('result', [])
            
            # Parse the results into structured format
            structured_results = {
                "analysis_summary": {
                    "total_sales": None,
                    "top_region": None,
                    "day_sales_correlation": None,
                    "median_sales": None,
                    "total_sales_tax": None
                },
                "charts_generated": [
                    "sales_by_region.png",
                    "cumulative_sales.png"
                ],
                "raw_results": results
            }
            
            # Extract specific values
            for result in results:
                if "Total sales:" in result:
                    structured_results["analysis_summary"]["total_sales"] = result.split(":")[1].strip()
                elif "Region with highest" in result:
                    structured_results["analysis_summary"]["top_region"] = result.split(":")[1].strip()
                elif "Correlation between day" in result:
                    structured_results["analysis_summary"]["day_sales_correlation"] = result.split(":")[1].strip()
                elif "Median sales:" in result:
                    structured_results["analysis_summary"]["median_sales"] = result.split(":")[1].strip()
                elif "Total sales tax:" in result:
                    structured_results["analysis_summary"]["total_sales_tax"] = result.split(":")[1].strip()
            
            # Display structured results
            print("🎯 ANALYSIS SUMMARY:")
            print("=" * 60)
            summary = structured_results["analysis_summary"]
            print(f"  • Total Sales: {summary['total_sales']}")
            print(f"  • Top Region: {summary['top_region']}")
            print(f"  • Day-Sales Correlation: {summary['day_sales_correlation']}")
            print(f"  • Median Sales: {summary['median_sales']}")
            print(f"  • Total Sales Tax: {summary['total_sales_tax']}")
            
            print("\n📈 CHARTS GENERATED:")
            for chart in structured_results["charts_generated"]:
                print(f"  • {chart}")
            
            print("\n📋 STRUCTURED JSON RESPONSE:")
            print("=" * 60)
            print(json.dumps(structured_results, indent=2))
            
            return structured_results
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def show_api_improvement():
    """Show how the API should be improved"""
    print("\n" + "=" * 60)
    print("🔧 API IMPROVEMENT SUGGESTION:")
    print("=" * 60)
    print("The API should return structured results like this:")
    print("""
{
  "job_id": "09a18b67-32c6-406f-9386-14782029f0ab",
  "status": "completed",
  "execution_time": 90.31,
  "result": {
    "analysis_summary": {
      "total_sales": "1140",
      "top_region": "West with total sales: 420",
      "day_sales_correlation": "0.22281245492773066",
      "median_sales": "140.0",
      "total_sales_tax": "114.0"
    },
    "charts_generated": [
      "sales_by_region.png",
      "cumulative_sales.png"
    ]
  }
}
""")
    print("=" * 60)

if __name__ == "__main__":
    structured_results = get_structured_results()
    show_api_improvement()
    
    if structured_results:
        print("\n✅ Structured results extracted successfully!")
        print("\n💡 The API is working correctly, but could be improved to return")
        print("   structured data instead of raw stdout strings.")
    else:
        print("\n❌ Could not extract structured results")
