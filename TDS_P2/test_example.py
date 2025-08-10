#!/usr/bin/env python3
"""
Example test script to demonstrate the API functionality
"""

import requests
import json
import time
import tempfile
import os

API_URL = "http://localhost:8000"

def test_wikipedia_films_analysis():
    """Test the Wikipedia films analysis example"""
    
    # Create questions file
    questions_content = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes."""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(questions_content)
        questions_file = f.name

    try:
        print("Testing Data Analyst Agent API...")
        print(f"API URL: {API_URL}")
        
        # Check API health
        print("\n1. Checking API health...")
        health_response = requests.get(f"{API_URL}/health")
        print(f"Health check: {health_response.status_code}")
        print(f"Response: {health_response.json()}")
        
        # Submit analysis request
        print("\n2. Submitting analysis request...")
        
        files = {
            'questions': ('questions.txt', open(questions_file, 'rb'), 'text/plain')
        }
        
        response = requests.post(f"{API_URL}/api/", files=files)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        result = response.json()
        job_id = result['job_id']
        print(f"Job submitted successfully. Job ID: {job_id}")
        print(f"Status: {result['status']}")
        
        # Poll for results
        print("\n3. Waiting for results...")
        max_wait_time = 200  # seconds
        poll_interval = 5
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            status_response = requests.get(f"{API_URL}/api/job/{job_id}")
            
            if status_response.status_code != 200:
                print(f"Error checking status: {status_response.status_code}")
                break
            
            status_result = status_response.json()
            current_status = status_result['status']
            
            print(f"Status: {current_status}")
            
            if current_status == 'completed':
                print("\n4. Analysis completed!")
                print(f"Execution time: {status_result.get('execution_time', 'unknown')}s")
                
                results = status_result.get('result', [])
                print(f"\nResults ({len(results)} items):")
                
                for i, result_item in enumerate(results, 1):
                    if isinstance(result_item, str) and result_item.startswith('data:image/'):
                        print(f"{i}. [Base64 Image Data - {len(result_item)} characters]")
                    else:
                        print(f"{i}. {result_item}")
                
                break
            
            elif current_status == 'failed':
                print(f"\n4. Analysis failed!")
                error = status_result.get('error', 'Unknown error')
                print(f"Error: {error}")
                break
            
            else:
                print(f"Still processing... ({current_status})")
                time.sleep(poll_interval)
        
        else:
            print(f"\nTimeout: Analysis did not complete within {max_wait_time} seconds")
    
    finally:
        # Cleanup
        try:
            os.unlink(questions_file)
        except:
            pass

def test_simple_data_analysis():
    """Test with a simple CSV analysis"""
    
    # Create a sample CSV file
    csv_content = """name,age,salary,department
John,25,50000,Engineering
Jane,30,60000,Marketing
Bob,35,55000,Engineering
Alice,28,52000,Marketing
Charlie,32,58000,Engineering
Diana,26,51000,Marketing"""

    questions_content = """Analyze the employee data and answer:
1. What is the average salary by department?
2. What is the age distribution?
3. Create a bar chart showing average salary by department.

Return results as a JSON array of strings."""

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(questions_content)
        questions_file = f.name

    try:
        print("\n" + "="*50)
        print("Testing Simple Data Analysis...")
        
        files = {
            'questions': ('questions.txt', open(questions_file, 'rb'), 'text/plain'),
            'files': ('data.csv', open(csv_file, 'rb'), 'text/csv')
        }
        
        response = requests.post(f"{API_URL}/api/", files=files)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        result = response.json()
        job_id = result['job_id']
        print(f"Job ID: {job_id}")
        
        # Wait for completion
        for _ in range(40):  # 2 minutes max
            time.sleep(3)
            status_response = requests.get(f"{API_URL}/api/job/{job_id}")
            status_result = status_response.json()
            
            if status_result['status'] == 'completed':
                print("Analysis completed!")
                results = status_result.get('result', [])
                for i, result_item in enumerate(results, 1):
                    print(f"{i}. {result_item}")
                break
            elif status_result['status'] == 'failed':
                print(f"Analysis failed: {status_result.get('error')}")
                break
            else:
                print(f"Status: {status_result['status']}")
        
    finally:
        # Cleanup
        try:
            os.unlink(csv_file)
            os.unlink(questions_file)
        except:
            pass

if __name__ == "__main__":
    try:
        # Test 1: Wikipedia analysis
        test_wikipedia_films_analysis()
        
        # Test 2: Simple CSV analysis
        test_simple_data_analysis()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
