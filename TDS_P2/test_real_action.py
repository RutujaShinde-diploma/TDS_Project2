#!/usr/bin/env python3
"""
Test real action execution through the orchestrator to see why it fails
"""

import asyncio
import tempfile
from pathlib import Path

from models import Action, ActionType, ExecutionContext
from orchestrator import Orchestrator
from utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_real_orchestrator_execution():
    """Test exactly what the orchestrator does during real execution"""
    
    print("🔬 Testing Real Orchestrator Action Execution")
    print("=" * 60)
    
    # Create a workspace like the real system
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = temp_dir
        
        # Create the exact scraping action from the plan
        scrape_action = Action(
            action_id="action_001",
            type=ActionType.SCRAPE,
            description="Scrape Wikipedia highest grossing films data",
            parameters={
                "url": "https://en.wikipedia.org/wiki/List_of_highest-grossing_films",
                "target": "table"
            },
            output_files=["highest_grossing_films.csv"],
            estimated_time=60
        )
        
        print(f"📂 Workspace: {workspace_path}")
        print(f"🎯 Action: {scrape_action.action_id}")
        print(f"🌐 URL: {scrape_action.parameters['url']}")
        print()
        
        # Create context like the orchestrator does
        context = ExecutionContext(
            workspace_path=workspace_path,
            available_files=[]
        )
        
        # Use the real orchestrator
        orchestrator = Orchestrator()
        
        print("🎭 Executing action through real orchestrator...")
        
        try:
            # Execute the action exactly like the orchestrator does
            action_result = await orchestrator._execute_action(scrape_action, context)
            
            print(f"📊 Status: {action_result.status}")
            print(f"⏱️ Execution time: {action_result.execution_time}")
            
            if action_result.status.value == "completed":
                print("✅ Action completed successfully")
                print(f"📤 Output success: {action_result.output.get('success', 'Unknown')}")
                print(f"📋 Stdout: {action_result.output.get('stdout', 'No stdout')[:300]}...")
                print(f"📋 Stderr: {action_result.output.get('stderr', 'No stderr')}")
                
                # Check files
                workspace = Path(workspace_path)
                csv_files = list(workspace.glob("*.csv"))
                all_files = list(workspace.glob("*"))
                
                print(f"📁 CSV files created: {[f.name for f in csv_files]}")
                print(f"📁 All files created: {[f.name for f in all_files]}")
                
                if csv_files:
                    # Check CSV content
                    import pandas as pd
                    try:
                        df = pd.read_csv(csv_files[0])
                        print(f"📊 CSV shape: {df.shape}")
                        print(f"📊 CSV columns: {list(df.columns)}")
                        print(f"📊 First few rows:")
                        print(df.head(2))
                    except Exception as e:
                        print(f"❌ Failed to read CSV: {e}")
                
            else:
                print("❌ Action failed")
                print(f"💥 Error: {action_result.error}")
                print(f"📋 Retry count: {action_result.retry_count}")
                
                if action_result.output:
                    print(f"📤 Output: {action_result.output}")
                
                if action_result.generated_code:
                    print("📝 Generated code:")
                    print("-" * 30)
                    print(action_result.generated_code[:500])
                    print("-" * 30)
        
        except Exception as e:
            print(f"💥 Exception during execution: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_real_orchestrator_execution())
