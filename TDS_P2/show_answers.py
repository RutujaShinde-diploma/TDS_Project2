import json
import requests

def get_analysis_answers(job_id="09a18b67-32c6-406f-9386-14782029f0ab"):
    """Get the actual analysis answers from the job results"""
    
    print("📊 EXTRACTING ANALYSIS ANSWERS")
    print("=" * 60)
    
    try:
        # Method 1: Try API endpoint first
        response = requests.get(f'http://localhost:8000/api/job/{job_id}')
        if response.status_code == 200:
            result = response.json()
            if result['status'] == 'completed':
                results = result.get('result', [])
                
                print("🎯 YOUR ANALYSIS ANSWERS:")
                print("=" * 60)
                
                # Extract the actual answers from the results
                answers = []
                for i, result_item in enumerate(results):
                    if "Total sales:" in result_item:
                        answers.append(f"1. Total Sales: {result_item.split(':')[1].strip()}")
                    elif "Region with highest" in result_item:
                        answers.append(f"2. Top Region: {result_item.split(':')[1].strip()}")
                    elif "Correlation between day" in result_item:
                        answers.append(f"3. Day-Sales Correlation: {result_item.split(':')[1].strip()}")
                    elif "Median sales:" in result_item:
                        answers.append(f"4. Median Sales: {result_item.split(':')[1].strip()}")
                    elif "Total sales tax:" in result_item:
                        answers.append(f"5. Total Sales Tax: {result_item.split(':')[1].strip()}")
                
                # Display answers nicely
                for answer in answers:
                    print(f"  {answer}")
                
                print("\n📈 CHARTS GENERATED:")
                print("  • Bar Chart: sales_by_region.png")
                print("  • Cumulative Chart: cumulative_sales.png")
                
                return answers
        
        # Method 2: Read from job storage file
        print("📂 Reading from job storage...")
        with open(f'job_storage/{job_id}.json', 'r') as f:
            job_data = json.load(f)
        
        if job_data['status'] == 'completed':
            results = job_data.get('result', [])
            
            print("🎯 YOUR ANALYSIS ANSWERS:")
            print("=" * 60)
            
            # Extract the actual answers from the results
            answers = []
            for i, result_item in enumerate(results):
                if "Total sales:" in result_item:
                    answers.append(f"1. Total Sales: {result_item.split(':')[1].strip()}")
                elif "Region with highest" in result_item:
                    answers.append(f"2. Top Region: {result_item.split(':')[1].strip()}")
                elif "Correlation between day" in result_item:
                    answers.append(f"3. Day-Sales Correlation: {result_item.split(':')[1].strip()}")
                elif "Median sales:" in result_item:
                    answers.append(f"4. Median Sales: {result_item.split(':')[1].strip()}")
                elif "Total sales tax:" in result_item:
                    answers.append(f"5. Total Sales Tax: {result_item.split(':')[1].strip()}")
            
            # Display answers nicely
            for answer in answers:
                print(f"  {answer}")
            
            print("\n📈 CHARTS GENERATED:")
            print("  • Bar Chart: sales_by_region.png")
            print("  • Cumulative Chart: cumulative_sales.png")
            
            return answers
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def show_raw_results(job_id="09a18b67-32c6-406f-9386-14782029f0ab"):
    """Show the raw results from the job"""
    
    print("\n📋 RAW RESULTS FROM JOB:")
    print("=" * 60)
    
    try:
        with open(f'job_storage/{job_id}.json', 'r') as f:
            job_data = json.load(f)
        
        results = job_data.get('result', [])
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    answers = get_analysis_answers()
    
    if answers:
        print("\n✅ Analysis answers extracted successfully!")
        print("\n💡 To get answers directly in API response, the system needs to be modified")
        print("   to return the actual analysis results instead of just the job status.")
    else:
        print("\n❌ Could not extract answers")
    
    # Show raw results for debugging
    show_raw_results()
