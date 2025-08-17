#!/usr/bin/env python3
"""
Direct API test script to verify our endpoint works correctly.
This simulates what the test framework should be doing.
"""

import requests
import json

def test_api_directly():
    """Test our API endpoint directly"""
    api_url = "https://tds-data-analyst-agent-9eb3.onrender.com/api/"
    
    # Test questions content
    questions_content = """Analyze this simple dataset:

Name,Age,Salary
John,25,50000
Alice,30,65000
Bob,35,75000
Carol,28,55000

Answer these questions:
1. What is the average age?
2. What is the total salary?
3. Who has the highest salary?"""
    
    # Prepare the request
    files = {
        'questions': ('questions.txt', questions_content, 'text/plain')
    }
    
    print(f"Testing API endpoint: {api_url}")
    print(f"Questions content length: {len(questions_content)}")
    print("Sending request...")
    
    try:
        # Make the POST request
        response = requests.post(api_url, files=files, timeout=300)
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! API responded with:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ ERROR! API returned status {response.status_code}")
            print(f"Response text: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: API took too long to respond")
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Could not connect to API")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {str(e)}")

if __name__ == "__main__":
    test_api_directly()
