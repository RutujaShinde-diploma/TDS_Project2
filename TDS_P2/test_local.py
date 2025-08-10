#!/usr/bin/env python3
"""
Quick local test script - simpler than test_example.py
"""

import requests
import json
import time
import tempfile
import os

API_URL = "http://localhost:8000"

def test_api_health():
    """Test if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Make sure the API is running: python start_local.py")
        return False

def test_simple_analysis():
    """Test with a simple analysis task"""
    
    # Create a simple question
    questions_content = """Analyze the following data and provide answers:

Create a simple dataset with 5 rows of sample employee data (name, age, salary).
Then answer:
1. What is the average salary?
2. What is the age range?
3. Create a simple summary

Return results as a JSON array of strings."""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(questions_content)
        questions_file = f.name

    try:
        print("\n🧪 Testing simple analysis...")
        
        # Submit request
        with open(questions_file, 'rb') as f:
            files = {'questions': ('questions.txt', f, 'text/plain')}
            response = requests.post(f"{API_URL}/api/", files=files)
        
        if response.status_code != 200:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        result = response.json()
        job_id = result.get('job_id')
        print(f"✅ Job submitted: {job_id}")
        
        # Poll for results
        print("⏳ Waiting for results...")
        for i in range(30):  # 90 seconds max
            time.sleep(3)
            
            status_response = requests.get(f"{API_URL}/api/job/{job_id}")
            if status_response.status_code != 200:
                print(f"❌ Status check failed: {status_response.status_code}")
                break
            
            status_result = status_response.json()
            current_status = status_result.get('status')
            
            print(f"Status: {current_status}")
            
            if current_status == 'completed':
                print("✅ Analysis completed!")
                results = status_result.get('result', [])
                print(f"\n📊 Results ({len(results)} items):")
                for i, result_item in enumerate(results, 1):
                    print(f"{i}. {result_item}")
                return True
            
            elif current_status == 'failed':
                print("❌ Analysis failed!")
                error = status_result.get('error', 'Unknown error')
                print(f"Error: {error}")
                return False
        
        print("⏰ Timeout waiting for results")
        return False
        
    finally:
        # Cleanup
        try:
            os.unlink(questions_file)
        except:
            pass

def test_csv_analysis():
    """Test with actual CSV data"""
    
    # Create sample CSV
    csv_content = """name,age,salary,department
Alice,25,50000,Engineering
Bob,30,60000,Marketing
Charlie,35,55000,Engineering
Diana,28,52000,Marketing
Eve,32,58000,Engineering"""

    questions_content = """Analyze the employee CSV data and answer:
1. How many employees are there?
2. What is the average salary by department?
3. What is the overall average age?

Return results as a JSON array of strings."""

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(questions_content)
        questions_file = f.name

    try:
        print("\n📊 Testing CSV analysis...")
        
        # Submit request with files
        files = {
            'questions': ('questions.txt', open(questions_file, 'rb'), 'text/plain'),
            'files': ('employees.csv', open(csv_file, 'rb'), 'text/csv')
        }
        
        response = requests.post(f"{API_URL}/api/", files=files)
        
        # Close files
        files['questions'][1].close()
        files['files'][1].close()
        
        if response.status_code != 200:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        result = response.json()
        job_id = result.get('job_id')
        print(f"✅ Job submitted: {job_id}")
        
        # Poll for results
        print("⏳ Waiting for results...")
        for i in range(30):
            time.sleep(3)
            
            status_response = requests.get(f"{API_URL}/api/job/{job_id}")
            status_result = status_response.json()
            current_status = status_result.get('status')
            
            print(f"Status: {current_status}")
            
            if current_status == 'completed':
                print("✅ CSV analysis completed!")
                results = status_result.get('result', [])
                print(f"\n📊 Results:")
                for i, result_item in enumerate(results, 1):
                    print(f"{i}. {result_item}")
                return True
            
            elif current_status == 'failed':
                print("❌ CSV analysis failed!")
                error = status_result.get('error', 'Unknown error')
                print(f"Error: {error}")
                return False
        
        print("⏰ Timeout waiting for results")
        return False
        
    finally:
        # Cleanup
        try:
            os.unlink(csv_file)
            os.unlink(questions_file)
        except:
            pass

def main():
    """Run all tests"""
    print("🧪 Data Analyst Agent API - Local Testing")
    print("=" * 50)
    
    # Test 1: API Health
    print("\n1. Testing API health...")
    if not test_api_health():
        print("\n❌ API is not running. Please start it first:")
        print("   python start_local.py")
        return
    
    # Test 2: Simple analysis
    print("\n2. Testing simple analysis...")
    test_simple_analysis()
    
    # Test 3: CSV analysis
    print("\n3. Testing CSV analysis...")
    test_csv_analysis()
    
    print("\n" + "=" * 50)
    print("🎉 Local testing completed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
