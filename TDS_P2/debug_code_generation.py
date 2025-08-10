#!/usr/bin/env python3
"""
Debug script to see what code is generated for SQL actions
"""

import asyncio
import tempfile

# Import the components we need to test
from code_generator import CodeGenerator
from models import Action, ActionType, ExecutionContext

async def test_code_generation():
    """Test code generation for SQL actions"""
    
    # Create test actions
    action_003 = Action(
        action_id="action_003",
        type=ActionType.SQL,
        description="Query to find the number of $2 bn movies released before 2000",
        parameters={
            "query": "SELECT COUNT(*) FROM data WHERE Gross_Revenue >= 2000000000 AND Year < 2000"
        },
        input_files=[],
        output_files=["query_result_1.json"],
        input_variables=[],
        output_variables=[],
        dependencies=["action_002"],
        estimated_time=30
    )
    
    action_004 = Action(
        action_id="action_004",
        type=ActionType.SQL,
        description="Query to find the earliest film that grossed over $1.5 bn",
        parameters={
            "query": "SELECT Title FROM data WHERE Gross_Revenue >= 1500000000 ORDER BY Year ASC LIMIT 1"
        },
        input_files=[],
        output_files=["query_result_2.json"],
        input_variables=[],
        output_variables=[],
        dependencies=["action_002"],
        estimated_time=30
    )
    
    # Create workspace
    workspace_path = tempfile.mkdtemp()
    print(f"Workspace: {workspace_path}")
    
    # Create context
    context = ExecutionContext(
        workspace_path=workspace_path,
        available_files=["highest_grossing_films.csv"]
    )
    
    # Generate code
    code_generator = CodeGenerator()
    
    print("\n" + "="*50)
    print("ACTION 003 - COUNT QUERY")
    print("="*50)
    code_003 = await code_generator.generate_code(action_003, context)
    print(f"Generated code:\n{code_003}")
    
    print("\n" + "="*50)
    print("ACTION 004 - SELECT QUERY")
    print("="*50)
    code_004 = await code_generator.generate_code(action_004, context)
    print(f"Generated code:\n{code_004}")

if __name__ == "__main__":
    asyncio.run(test_code_generation())
