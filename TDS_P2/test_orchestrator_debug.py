#!/usr/bin/env python3
"""
Debug test for orchestrator to understand why web scraping actions fail
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

from models import Action, ActionType, ExecutionPlan, ExecutionContext
from orchestrator import Orchestrator
from sandbox import SandboxExecutor
from code_generator import CodeGenerator
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

async def test_simple_scrape_action():
    """Test a simple Wikipedia scraping action to debug issues"""
    
    print("🔍 Testing Simple Wikipedia Scraping Action")
    print("=" * 60)
    
    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = temp_dir
        
        # Create a simple scraping action
        scrape_action = Action(
            action_id="test_scrape_001",
            type=ActionType.SCRAPE,
            description="Test Wikipedia scraping",
            parameters={
                "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
                "target": "table"
            },
            output_files=["films_data.csv"],
            estimated_time=30
        )
        
        print(f"📂 Workspace: {workspace_path}")
        print(f"🎯 Action: {scrape_action.action_id}")
        print(f"🌐 URL: {scrape_action.parameters['url']}")
        print()
        
        # Test individual components
        await test_code_generation(scrape_action, workspace_path)
        await test_sandbox_execution(scrape_action, workspace_path)
        await test_full_orchestrator(scrape_action, workspace_path)

async def test_code_generation(action: Action, workspace_path: str):
    """Test code generation for the scraping action"""
    print("1️⃣ Testing Code Generation...")
    
    try:
        code_generator = CodeGenerator()
        context = ExecutionContext(
            workspace_path=workspace_path,
            available_files=[]
        )
        
        # Generate code
        generated_code = await code_generator.generate_code(action, context)
        
        print("✅ Code generation successful")
        print(f"📝 Generated code length: {len(generated_code)} chars")
        print("📋 Code preview:")
        print("-" * 40)
        print(generated_code[:500] + "..." if len(generated_code) > 500 else generated_code)
        print("-" * 40)
        print()
        
        return generated_code
        
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
        print()
        return None

async def test_sandbox_execution(action: Action, workspace_path: str):
    """Test sandbox execution with pre-generated code"""
    print("2️⃣ Testing Sandbox Execution...")
    
    try:
        # Simple test code for scraping
        test_code = '''
import requests
import pandas as pd
from bs4 import BeautifulSoup
import sys

print("Starting Wikipedia scraping test...")

try:
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    print(f"Fetching: {url}")
    
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    print(f"Page fetched successfully ({len(response.content)} bytes)")
    
    # Try to parse tables
    tables = pd.read_html(url, header=0)
    print(f"Found {len(tables)} tables")
    
    if tables:
        df = tables[0]  # First table
        print(f"First table shape: {df.shape}")
        print("First few rows:")
        print(df.head(3))
        
        # Save to CSV
        output_file = "films_data.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved data to {output_file}")
        
    print("Scraping test completed successfully!")
    
except Exception as e:
    print(f"Scraping failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        
        sandbox = SandboxExecutor()
        
        print("Executing test code in sandbox...")
        result = await sandbox.execute_code(test_code, workspace_path, action.action_id)
        
        if result["success"]:
            print("SUCCESS: Sandbox execution successful")
            print(f"Output: {result.get('output', 'No output')}")
            print(f"Execution time: {result.get('execution_time', 'Unknown')}s")
            
            # Check if files were created
            workspace = Path(workspace_path)
            files = list(workspace.glob("*.csv"))
            if files:
                print(f"Files created: {[f.name for f in files]}")
            else:
                print("WARNING: No CSV files were created")
        else:
            print("FAILED: Sandbox execution failed")
            print(f"Error: {result.get('error', 'Unknown error')}")
            print(f"Output: {result.get('output', 'No output')}")
        
        print()
        return result
        
    except Exception as e:
        print(f"❌ Sandbox test failed: {e}")
        print()
        return None

async def test_full_orchestrator(action: Action, workspace_path: str):
    """Test full orchestrator execution"""
    print("3️⃣ Testing Full Orchestrator...")
    
    try:
        orchestrator = Orchestrator()
        context = ExecutionContext(
            workspace_path=workspace_path,
            available_files=[]
        )
        
        print("🎭 Executing action through orchestrator...")
        action_result = await orchestrator._execute_action(action, context)
        
        print(f"📊 Action result status: {action_result.status}")
        print(f"⏱️ Execution time: {action_result.execution_time}")
        
        if action_result.status.value == "completed":
            print("✅ Orchestrator execution successful")
            print(f"📤 Output keys: {list(action_result.output.keys()) if action_result.output else 'None'}")
        else:
            print("❌ Orchestrator execution failed")
            print(f"💥 Error: {action_result.error}")
            if action_result.generated_code:
                print("📋 Generated code:")
                print("-" * 30)
                print(action_result.generated_code[:300] + "...")
                print("-" * 30)
        
        print()
        return action_result
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return None

async def test_dependencies():
    """Test if required dependencies are available"""
    print("0️⃣ Testing Dependencies...")
    
    dependencies = [
        ("requests", "HTTP requests"),
        ("beautifulsoup4", "HTML parsing"),
        ("pandas", "Data manipulation"),
        ("lxml", "XML/HTML parser for pandas")
    ]
    
    for package, description in dependencies:
        try:
            if package == "beautifulsoup4":
                import bs4
                print(f"✅ {package} ({description}) - version {bs4.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")
                print(f"✅ {package} ({description}) - version {version}")
        except ImportError:
            print(f"❌ {package} ({description}) - NOT AVAILABLE")
    
    print()

async def main():
    """Run all orchestrator debug tests"""
    print("🐛 Orchestrator Debug Test Suite")
    print("=" * 60)
    print()
    
    # Test dependencies first
    await test_dependencies()
    
    # Test orchestrator
    await test_simple_scrape_action()
    
    print("🏁 Debug tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
