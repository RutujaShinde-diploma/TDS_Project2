#!/usr/bin/env python3
"""
Debug script to see what code is generated for the export action
"""

import asyncio
import tempfile

# Import the components we need to test
from code_generator import CodeGenerator
from models import Action, ActionType, ExecutionContext

async def test_export_generation():
    """Test code generation for export action"""
    
    # Create test action
    action_005 = Action(
        action_id="action_005",
        type=ActionType.EXPORT,
        description="Format and export results",
        parameters={
            "input_files": ["query_result_1.json", "query_result_2.json"],
            "format": "json"
        },
        input_files=[],
        output_files=["final_results.json"],
        input_variables=[],
        output_variables=[],
        dependencies=["action_003", "action_004"],
        estimated_time=20
    )
    
    # Create workspace
    workspace_path = tempfile.mkdtemp()
    print(f"Workspace: {workspace_path}")
    
    # Create context with the input files
    context = ExecutionContext(
        workspace_path=workspace_path,
        available_files=["query_result_1.json", "query_result_2.json"]
    )
    
    # Generate code
    code_generator = CodeGenerator()
    
    print("\n" + "="*50)
    print("ACTION 005 - EXPORT")
    print("="*50)
    code_005 = await code_generator.generate_code(action_005, context)
    print(f"Generated code:\n{code_005}")

if __name__ == "__main__":
    asyncio.run(test_export_generation())
