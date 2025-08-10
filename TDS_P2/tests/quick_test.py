#!/usr/bin/env python3
"""
Quick Test Suite - Essential tests for rapid feedback
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def quick_unit_tests():
    """Essential unit tests"""
    print("🔧 Quick Unit Tests")
    print("-" * 30)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Imports
    total_tests += 1
    try:
        from config import config
        from models import Action, ActionType
        from sandbox import CodeValidator
        print("✅ Imports working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Import test failed: {e}")
    
    # Test 2: Code Validation
    total_tests += 1
    try:
        validator = CodeValidator()
        safe_code = "print('hello')"
        is_safe, _ = validator.validate_code(safe_code)
        assert is_safe
        print("✅ Code validator working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Code validator test failed: {e}")
    
    # Test 3: Models
    total_tests += 1
    try:
        action = Action(
            action_id="test",
            type=ActionType.STATS,
            description="Test"
        )
        assert action.action_id == "test"
        print("✅ Models working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Model test failed: {e}")
    
    return tests_passed, total_tests

async def quick_api_tests():
    """Essential API tests"""
    print("\n🌐 Quick API Tests")
    print("-" * 30)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Health Check
    total_tests += 1
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        assert response.status_code == 200
        print("✅ API health check working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test 2: Job Submission
    total_tests += 1
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Simple test: What is 1 + 1?")
            question_file = f.name
        
        try:
            files = {'questions': ('questions.txt', open(question_file, 'rb'), 'text/plain')}
            response = requests.post("http://localhost:8000/api/", files=files, timeout=10)
            files['questions'][1].close()
            
            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            print(f"✅ Job submission working (job: {data['job_id'][:8]}...)")
            tests_passed += 1
        finally:
            os.unlink(question_file)
            
    except Exception as e:
        print(f"❌ Job submission failed: {e}")
    
    return tests_passed, total_tests

async def quick_execution_test():
    """Test basic code execution"""
    print("\n⚡ Quick Execution Test")
    print("-" * 30)
    
    try:
        from sandbox import SandboxExecutor
        
        sandbox = SandboxExecutor()
        test_code = """
import pandas as pd
data = {'numbers': [1, 2, 3, 4, 5]}
df = pd.DataFrame(data)
result = df['numbers'].mean()
print(f"Mean: {result}")
"""
        
        workspace = project_root / "workspace"
        workspace.mkdir(exist_ok=True)
        
        result = await sandbox.execute_code(test_code, str(workspace), "quick_test")
        
        if result["success"] and "Mean: 3.0" in result.get("stdout", ""):
            print("✅ Code execution working")
            return 1, 1
        else:
            print(f"❌ Code execution failed: {result.get('error', 'Unknown error')}")
            return 0, 1
            
    except Exception as e:
        print(f"❌ Execution test crashed: {e}")
        return 0, 1

async def main():
    """Run quick tests"""
    start_time = time.time()
    
    print("🚀 Quick Test Suite - Data Analyst Agent API")
    print("=" * 50)
    
    # Run tests
    unit_passed, unit_total = await quick_unit_tests()
    api_passed, api_total = await quick_api_tests()
    exec_passed, exec_total = await quick_execution_test()
    
    # Summary
    total_passed = unit_passed + api_passed + exec_passed
    total_tests = unit_total + api_total + exec_total
    duration = time.time() - start_time
    
    print(f"\n📊 Quick Test Summary")
    print("-" * 30)
    print(f"Passed: {total_passed}/{total_tests}")
    print(f"Duration: {duration:.2f}s")
    
    if total_passed == total_tests:
        print("🎉 All quick tests passed! System is operational.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the details above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
