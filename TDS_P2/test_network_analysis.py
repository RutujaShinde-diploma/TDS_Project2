#!/usr/bin/env python3
"""
Test script for network analysis functionality
"""

import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from models import JobRequest, Action, ActionType, ExecutionContext
from planner import PlannerModule
from code_generator import CodeGenerator
from orchestrator import Orchestrator
from sandbox import SandboxExecutor
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

async def test_network_analysis():
    """Test the network analysis workflow"""
    
    # Create test workspace
    test_workspace = Path("test_network_workspace")
    if test_workspace.exists():
        shutil.rmtree(test_workspace)
    test_workspace.mkdir()
    
    # Copy test files
    shutil.copy("test2/edges.csv", test_workspace / "edges.csv")
    shutil.copy("test2/questions.txt", test_workspace / "questions.txt")
    
    # Create job request
    job_request = JobRequest(
        questions="How many edges are in the network?\nWhat is the highest degree node?\nWhat is the average degree?\nWhat is the network density?\nWhat is the shortest path between Alice and Eve?\nGenerate a network graph visualization\nGenerate a degree histogram",
        files=["edges.csv"]
    )
    
    # Create execution context
    context = ExecutionContext(
        workspace_path=str(test_workspace),
        available_files=["edges.csv", "questions.txt"],
        variables={}
    )
    
    try:
        # Test planner
        logger.info("🔍 Testing planner...")
        planner = PlannerModule()
        plan = await planner.create_plan(job_request, str(test_workspace))
        logger.info(f"✅ Plan created: {len(plan.actions)} actions")
        
        for action in plan.actions:
            logger.info(f"  - {action.action_id}: {action.type.value} -> {action.output_files}")
        
        # Test code generation
        logger.info("🔍 Testing code generation...")
        code_generator = CodeGenerator()
        
        for action in plan.actions:
            logger.info(f"🔍 Generating code for {action.action_id} ({action.type.value})")
            code = await code_generator.generate_code(action, context)
            logger.info(f"✅ Code generated for {action.action_id}: {len(code)} chars")
            
            # Save generated code for inspection
            code_file = test_workspace / f"{action.action_id}_code.py"
            with open(code_file, 'w') as f:
                f.write(code)
            logger.info(f"💾 Code saved to {code_file}")
        
        # Test orchestrator
        logger.info("🔍 Testing orchestrator...")
        orchestrator = Orchestrator()
        
        # Execute actions
        results = []
        for action in plan.actions:
            logger.info(f"🚀 Executing {action.action_id} ({action.type.value})")
            
            # Get code
            code = await orchestrator._get_action_code(action, context)
            
            # Execute in sandbox
            sandbox = SandboxExecutor()
            result = await sandbox.execute_code(code, str(test_workspace), action.action_id)
            results.append(result)
            
            logger.info(f"✅ {action.action_id} completed: {result.status.value}")
            if result.stdout:
                logger.info(f"📤 Stdout: {result.stdout[:200]}...")
            if result.error:
                logger.error(f"❌ Error: {result.error}")
        
        # Test result assembly
        logger.info("🔍 Testing result assembly...")
        final_result = await orchestrator._assemble_final_result(results, context)
        logger.info(f"✅ Final result: {final_result}")
        
        # Check if final_results.json was created
        final_file = test_workspace / "final_results.json"
        if final_file.exists():
            with open(final_file, 'r') as f:
                final_data = json.load(f)
            logger.info(f"✅ Final results file: {final_data}")
        else:
            logger.error("❌ Final results file not created!")
        
        # List all files in workspace
        all_files = list(test_workspace.glob("*"))
        logger.info(f"📁 Files in workspace: {[f.name for f in all_files]}")
        
        return final_result
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
    finally:
        # Cleanup
        if test_workspace.exists():
            shutil.rmtree(test_workspace)

if __name__ == "__main__":
    asyncio.run(test_network_analysis())
