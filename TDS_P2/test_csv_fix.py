#!/usr/bin/env python3
"""
Test script to verify CSV analysis fix
"""

import asyncio
import sys
from pathlib import Path

async def test_csv_analysis():
    """Test the CSV analysis workflow"""
    print("🧪 Testing CSV analysis fix...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from models import JobRequest, ExecutionPlan, Action, ActionType
        from planner import PlannerModule
        from orchestrator import Orchestrator
        print("   ✅ All imports successful")
        
        # Test planner
        print("2. Testing planner...")
        planner = PlannerModule()
        
        # Create a test job request
        job_request = JobRequest(
            questions="Analyze sample-sales.csv. Return a JSON object with keys: total_sales: number",
            files=["sample-sales.csv"]
        )
        
        # Create workspace
        workspace_path = "test_workspace"
        Path(workspace_path).mkdir(exist_ok=True)
        
        # Copy test files
        import shutil
        if Path("test_sample_sales/sample-sales.csv").exists():
            shutil.copy("test_sample_sales/sample-sales.csv", workspace_path)
        if Path("test_sample_sales/questions.txt").exists():
            shutil.copy("test_sample_sales/questions.txt", workspace_path)
        
        print("   ✅ Test files copied")
        
        # Generate plan
        plan = await planner.create_plan(job_request, workspace_path)
        print(f"   ✅ Plan created with {len(plan.actions)} actions")
        print(f"   📋 Action types: {[action.type.value for action in plan.actions]}")
        
        # Test orchestrator
        print("3. Testing orchestrator...")
        orchestrator = Orchestrator()
        
        # Execute plan
        result = await orchestrator.execute_plan(plan, workspace_path)
        print(f"   ✅ Plan executed successfully")
        print(f"   📊 Result type: {type(result)}")
        print(f"   📊 Result: {result}")
        
        # Cleanup
        shutil.rmtree(workspace_path, ignore_errors=True)
        
        print("\n🎉 CSV analysis test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_csv_analysis())
    sys.exit(0 if success else 1)
