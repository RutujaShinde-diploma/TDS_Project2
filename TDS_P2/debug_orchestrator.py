#!/usr/bin/env python3
"""
Debug script to test orchestrator functionality
"""

import asyncio
import os
import tempfile
from pathlib import Path

# Import the components we need to test
from orchestrator import Orchestrator
from models import Action, ActionType, ExecutionPlan, ExecutionContext

async def test_orchestrator():
    """Test the orchestrator with the exact same plan as the failing job"""
    
    # Create the exact same plan as the failing job
    plan = ExecutionPlan(
        plan_id="highest_grossing_films_analysis",
        task_description="Scrape the list of highest grossing films from Wikipedia. It is at the URL:\nhttps://en.wikipedia.org/wiki/List_of_highest-grossing_films\n\nAnswer the following questions and respond with a JSON array of strings containing the answer.\n\n1. How many $2 bn movies were released before 2000?\n2. Which is the earliest film that grossed over $1.5 bn?",
        actions=[
            Action(
                action_id="action_001",
                type=ActionType.SCRAPE,
                description="Scrape Wikipedia highest grossing films data",
                parameters={
                    "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
                    "target": "table"
                },
                input_files=[],
                output_files=["highest_grossing_films.csv"],
                input_variables=[],
                output_variables=[],
                dependencies=[],
                estimated_time=60
            ),
            Action(
                action_id="action_002",
                type=ActionType.LOAD,
                description="Load scraped data into DataFrame",
                parameters={
                    "file": "highest_grossing_films.csv"
                },
                input_files=[],
                output_files=[],
                input_variables=[],
                output_variables=[],
                dependencies=["action_001"],
                estimated_time=10
            ),
            Action(
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
            ),
            Action(
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
            ),
            Action(
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
        ],
        estimated_total_time=150,
        metadata={}
    )
    
    # Create workspace
    workspace_path = tempfile.mkdtemp()
    print(f"Workspace: {workspace_path}")
    
    # Execute plan
    orchestrator = Orchestrator()
    print("Executing plan...")
    result = await orchestrator.execute_plan(plan, workspace_path)
    
    print(f"Plan execution result: {result}")
    
    # Check workspace
    workspace = Path(workspace_path)
    files = list(workspace.glob("*"))
    print(f"Files in workspace: {[f.name for f in files]}")

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
