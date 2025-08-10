#!/usr/bin/env python3
"""
Test the exact flow that the main API uses for job execution
"""

import asyncio
import tempfile
from pathlib import Path

from models import JobRequest, ExecutionPlan, Action, ActionType
from planner import PlannerModule
from validator import PlanValidator
from orchestrator import Orchestrator
from utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_exact_api_flow():
    """Test the exact same flow as the main API"""
    
    print("🔬 Testing Exact API Flow")
    print("=" * 50)
    
    # Create workspace like the API does
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = temp_dir
        
        # Create job request like the API does
        questions_text = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?"""
        
        job_request = JobRequest(
            questions=questions_text,
            files=[]
        )
        
        print(f"📂 Workspace: {workspace_path}")
        print(f"📋 Questions: {questions_text[:100]}...")
        print()
        
        try:
            # Step 1: Generate execution plan (like the API does)
            print("1️⃣ Generating execution plan...")
            planner = PlannerModule()
            plan = await planner.create_plan(job_request, workspace_path)
            print(f"✅ Plan generated with {len(plan.actions)} actions")
            
            # Step 2: Validate plan (like the API does)
            print("2️⃣ Validating plan...")
            validator = PlanValidator()
            validation_result = await validator.validate_plan(plan)
            
            if not validation_result.is_valid:
                print(f"❌ Plan validation failed: {validation_result.errors}")
                return
            print("✅ Plan validation passed")
            
            # Step 3: Execute plan (like the API does)
            print("3️⃣ Executing plan...")
            orchestrator = Orchestrator()
            
            # Add verbose logging
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
            
            result = await orchestrator.execute_plan(plan, workspace_path)
            
            print(f"✅ Plan execution completed")
            print(f"📊 Result type: {type(result)}")
            print(f"📊 Result content: {result}")
            
            # Check workspace files
            workspace = Path(workspace_path)
            all_files = list(workspace.glob("*"))
            csv_files = list(workspace.glob("*.csv"))
            json_files = list(workspace.glob("*.json"))
            
            print(f"📁 All files: {[f.name for f in all_files]}")
            print(f"📁 CSV files: {[f.name for f in csv_files]}")
            print(f"📁 JSON files: {[f.name for f in json_files]}")
            
            if csv_files:
                print("🎉 SUCCESS: CSV files were created!")
                # Try to read first CSV
                import pandas as pd
                df = pd.read_csv(csv_files[0])
                print(f"📊 CSV shape: {df.shape}")
                print(f"📊 CSV columns: {list(df.columns)}")
            else:
                print("❌ FAILED: No CSV files created")
                
        except Exception as e:
            print(f"💥 Error in flow: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_exact_api_flow())
