#!/usr/bin/env python3
"""
Test Orchestrator - Systematic testing with different levels
"""

import sys
import argparse
import asyncio
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_header(title: str):
    """Print formatted test header"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

async def run_quick_tests():
    """Run quick tests for rapid feedback"""
    print_header("QUICK TESTS - Essential Functionality")
    
    try:
        from tests.quick_test import main as quick_main
        return await quick_main()
    except Exception as e:
        print(f"❌ Quick tests failed to run: {e}")
        return 1

async def run_component_tests():
    """Run detailed component tests"""
    print_header("COMPONENT TESTS - Individual Components")
    
    try:
        from tests.component_tests import main as component_main
        return await component_main()
    except Exception as e:
        print(f"❌ Component tests failed to run: {e}")
        return 1

async def run_full_tests():
    """Run comprehensive test suite"""
    print_header("FULL TEST SUITE - Complete System Testing")
    
    try:
        from tests.test_runner import main as full_main
        return await full_main()
    except Exception as e:
        print(f"❌ Full tests failed to run: {e}")
        return 1

async def check_api_server():
    """Check if API server is running"""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=3)
        if response.status_code == 200:
            print("✅ API server is running")
            return True
        else:
            print(f"⚠️  API server responding with status {response.status_code}")
            return False
    except Exception:
        print("❌ API server is not running")
        print("   Start it with: python start_local.py")
        return False

async def main():
    """Main test orchestrator"""
    parser = argparse.ArgumentParser(description="Data Analyst Agent API Test Suite")
    parser.add_argument(
        "test_type", 
        choices=["quick", "component", "full", "all"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--no-api-check", 
        action="store_true",
        help="Skip API server check"
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true", 
        help="Continue running tests even if some fail"
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("🚀 Data Analyst Agent API - Test Orchestrator")
    print(f"Selected: {args.test_type.upper()} tests")
    
    # Check API server for API-dependent tests
    if not args.no_api_check and args.test_type in ["quick", "full", "all"]:
        print("\n🌐 Checking API server status...")
        if not await check_api_server():
            if args.test_type in ["full", "all"]:
                print("⚠️  Some tests require the API server to be running")
            else:
                print("❌ API tests require the server to be running")
                return 1
    
    # Run tests based on selection
    exit_codes = []
    
    if args.test_type == "quick":
        exit_codes.append(await run_quick_tests())
    
    elif args.test_type == "component":
        exit_codes.append(await run_component_tests())
    
    elif args.test_type == "full":
        exit_codes.append(await run_full_tests())
    
    elif args.test_type == "all":
        # Run all test types in sequence
        print_header("RUNNING ALL TEST SUITES")
        
        print("\n🚀 Phase 1: Quick Tests")
        quick_result = await run_quick_tests()
        exit_codes.append(quick_result)
        
        if quick_result != 0 and not args.continue_on_failure:
            print("❌ Quick tests failed. Stopping here (use --continue-on-failure to continue)")
        else:
            print("\n🚀 Phase 2: Component Tests")
            component_result = await run_component_tests()
            exit_codes.append(component_result)
            
            if component_result != 0 and not args.continue_on_failure:
                print("❌ Component tests failed. Stopping here (use --continue-on-failure to continue)")
            else:
                print("\n🚀 Phase 3: Full System Tests")
                full_result = await run_full_tests()
                exit_codes.append(full_result)
    
    # Final summary
    duration = time.time() - start_time
    
    print_header("TEST ORCHESTRATOR SUMMARY")
    
    total_suites = len(exit_codes)
    passed_suites = sum(1 for code in exit_codes if code == 0)
    
    print(f"📊 Test Suites: {passed_suites}/{total_suites} passed")
    print(f"⏱️  Total Duration: {duration:.2f}s")
    
    if passed_suites == total_suites:
        print("🎉 ALL TEST SUITES PASSED!")
        print("✅ System is ready for deployment")
        return 0
    else:
        print(f"⚠️  {total_suites - passed_suites} test suite(s) failed")
        print("❌ Review failed tests before deployment")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\n💥 Test orchestrator crashed: {e}")
        sys.exit(1)
