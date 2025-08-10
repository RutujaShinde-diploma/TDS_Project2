#!/usr/bin/env python3
"""
Test job status retrieval to verify the 500 error fix
"""

import requests
import time

API_URL = "http://localhost:8000"

def test_job_status():
    """Test retrieving the status of the recent job"""
    
    print("🔍 Testing Job Status Retrieval")
    print("=" * 50)
    
    # Use the recent job ID that we know completed
    job_id = "541174b5-bec6-4712-8d1a-55960944f48a"
    
    print(f"📋 Testing job: {job_id}")
    print()
    
    try:
        # Test API health first
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API server is healthy")
        else:
            print(f"⚠️ API health check returned: {health_response.status_code}")
        
        # Test job status retrieval
        print(f"🔍 Checking job status...")
        
        status_response = requests.get(f"{API_URL}/api/job/{job_id}", timeout=10)
        
        if status_response.status_code == 200:
            print("✅ Job status retrieved successfully!")
            
            result = status_response.json()
            print(f"📊 Status: {result.get('status')}")
            print(f"⏱️ Execution time: {result.get('execution_time', 'Unknown')}s")
            
            if result.get('result'):
                print(f"📋 Result: {result['result']}")
            
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
                
            if result.get('plan'):
                plan = result['plan']
                if isinstance(plan, dict):
                    print(f"📝 Plan actions: {len(plan.get('actions', []))}")
                else:
                    print(f"📝 Plan: {type(plan)}")
            
            return True
            
        elif status_response.status_code == 404:
            print("❌ Job not found (404)")
            return False
            
        elif status_response.status_code == 500:
            print("❌ Internal Server Error (500) - still happening!")
            print(f"Response: {status_response.text}")
            return False
            
        else:
            print(f"❌ Unexpected status code: {status_response.status_code}")
            print(f"Response: {status_response.text}")
            return False
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("Make sure the server is running: python start_local.py")
        return False
    
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def test_storage_stats():
    """Test the storage stats endpoint"""
    print("\n📊 Testing Storage Stats...")
    
    try:
        response = requests.get(f"{API_URL}/api/storage/stats", timeout=5)
        
        if response.status_code == 200:
            stats = response.json()
            print("✅ Storage stats retrieved:")
            print(f"   Total jobs: {stats.get('total_jobs', 0)}")
            print(f"   By status: {stats.get('by_status', {})}")
            print(f"   Storage size: {stats.get('storage_size_mb', 0)} MB")
            return True
        else:
            print(f"❌ Storage stats failed: {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Storage stats test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Job Status Test Suite")
    print("=" * 50)
    print()
    
    # Test job status
    status_works = test_job_status()
    
    # Test storage stats
    storage_works = test_storage_stats()
    
    print("\n📊 Test Summary")
    print("=" * 20)
    print(f"✅ Job status: {'PASS' if status_works else 'FAIL'}")
    print(f"✅ Storage stats: {'PASS' if storage_works else 'FAIL'}")
    
    if status_works and storage_works:
        print("\n🎉 All tests passed! 500 error fixed!")
    else:
        print("\n⚠️ Some tests failed - needs more debugging")
