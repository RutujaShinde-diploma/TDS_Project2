#!/usr/bin/env python3
"""
Debug script to test scraping functionality
"""

import asyncio
import os
import tempfile
from pathlib import Path

# Import the components we need to test
from sandbox import SandboxExecutor
from code_generator import CodeGenerator
from models import Action, ActionType, ExecutionContext

async def test_scraping():
    """Test the scraping action"""
    
    # Create a test action with the exact same parameters as the failing job
    action = Action(
        action_id="action_001",
        type=ActionType.SCRAPE,
        description="Scrape Wikipedia highest grossing films data",
        parameters={
            "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
            "target": "table"
        },
        input_files=[],
        output_files=["highest_grossing_films.csv"],  # This is the expected output file
        input_variables=[],
        output_variables=[],
        dependencies=[],
        estimated_time=60
    )
    
    # Create workspace
    workspace_path = tempfile.mkdtemp()
    print(f"Workspace: {workspace_path}")
    
    # Create context
    context = ExecutionContext(
        workspace_path=workspace_path,
        available_files=[]
    )
    
    # Generate code
    code_generator = CodeGenerator()
    print("Generating code...")
    code = await code_generator.generate_code(action, context)
    print(f"Generated code:\n{code}")
    
    # Execute code
    sandbox = SandboxExecutor()
    print("Executing code...")
    result = await sandbox.execute_code(code, workspace_path, action.action_id)
    
    print(f"Execution result: {result}")
    
    # Check workspace
    workspace = Path(workspace_path)
    files = list(workspace.glob("*"))
    print(f"Files in workspace: {[f.name for f in files]}")
    
    # Check if the expected output file was created
    expected_file = workspace / "highest_grossing_films.csv"
    if expected_file.exists():
        print(f"✅ Expected output file {expected_file.name} was created")
        print(f"File size: {expected_file.stat().st_size} bytes")
    else:
        print(f"❌ Expected output file {expected_file.name} was NOT created")

if __name__ == "__main__":
    asyncio.run(test_scraping())
