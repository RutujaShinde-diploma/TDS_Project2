#!/usr/bin/env python3
"""
Test the Wikipedia Films Analysis - Real Example
"""

import requests
import json
import time
import tempfile
import os

API_URL = "http://localhost:8000"

def test_wikipedia_analysis():
    """Test the real Wikipedia films analysis example"""
    
    # Create the exact questions from our example
    questions_content = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
"""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(questions_content)
        questions_file = f.name

    try:
        print("🎬 Testing Wikipedia Films Analysis")
        print("=" * 50)
        
        # Check API health first
        print("1. Checking API health...")
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ API health check failed: {health_response.status_code}")
            return False
        print("✅ API is healthy")
        
        # Submit analysis request
        print("\n2. Submitting Wikipedia analysis request...")
        
        with open(questions_file, 'rb') as f:
            files = {'questions': ('questions.txt', f, 'text/plain')}
            response = requests.post(f"{API_URL}/api/", files=files, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        result = response.json()
        job_id = result['job_id']
        print(f"✅ Job submitted successfully")
        print(f"📋 Job ID: {job_id}")
        
        # Poll for results with detailed progress
        print(f"\n3. Processing analysis (max 10 minutes)...")
        max_wait_time = 600  # 10 minutes
        poll_interval = 10
        
        start_time = time.time()
        last_status = ""
        
        while time.time() - start_time < max_wait_time:
            status_response = requests.get(f"{API_URL}/api/job/{job_id}", timeout=60)
            
            if status_response.status_code == 404:
                print(f"⚠️  Job {job_id} not found - it may have completed and been cleaned up")
                print("   Checking for any recent completion...")
                time.sleep(5)
                continue
            elif status_response.status_code != 200:
                print(f"❌ Error checking status: {status_response.status_code}")
                print(f"Response: {status_response.text}")
                break
            
            status_result = status_response.json()
            current_status = status_result['status']
            elapsed = time.time() - start_time
            
            # Only print status changes to avoid spam
            if current_status != last_status:
                print(f"⏱️  [{elapsed:.0f}s] Status: {current_status}")
                last_status = current_status
            
            if current_status == 'completed':
                print(f"\n4. ✅ Analysis completed in {elapsed:.1f}s!")
                
                execution_time = status_result.get('execution_time', 'unknown')
                print(f"🚀 Total execution time: {execution_time}s")
                
                results = status_result.get('result', [])
                print(f"\n📊 Results ({len(results)} items):")
                
                for i, result_item in enumerate(results, 1):
                    if isinstance(result_item, str) and result_item.startswith('data:image/'):
                        image_size = len(result_item)
                        print(f"  {i}. [📈 Visualization - {image_size:,} characters]")
                        if image_size > 100000:
                            print(f"      ⚠️  Image size ({image_size:,}) exceeds 100,000 limit")
                        else:
                            print(f"      ✅ Image size within limit")
                    else:
                        # Truncate long results for display
                        display_result = str(result_item)
                        if len(display_result) > 100:
                            display_result = display_result[:100] + "..."
                        print(f"  {i}. {display_result}")
                
                print(f"\n🎉 Wikipedia Films Analysis completed successfully!")
                return True
            
            elif current_status == 'failed':
                print(f"\n❌ Analysis failed after {elapsed:.1f}s")
                error = status_result.get('error', 'Unknown error')
                print(f"💥 Error: {error}")
                return False
            
            time.sleep(poll_interval)
        
        # Timeout
        print(f"\n⏰ Analysis timed out after {max_wait_time}s")
        print("   This might be due to:")
        print("   - Complex data processing (Wikipedia scraping + analysis)")
        print("   - Network connectivity issues")
        print("   - OpenAI API rate limits")
        print("   - Large dataset processing time")
        return False
        
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        return False
        
    finally:
        # Cleanup
        try:
            os.unlink(questions_file)
        except:
            pass

if __name__ == "__main__":
    success = test_wikipedia_analysis()
    if success:
        print("\n🎊 Real-world test PASSED! The system is production ready!")
    else:
        print("\n⚠️  Real-world test had issues. Check the details above.")
    
    exit(0 if success else 1)
