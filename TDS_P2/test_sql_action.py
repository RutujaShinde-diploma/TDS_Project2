#!/usr/bin/env python3
"""
Test SQL action specifically to see what code is generated and why it fails
"""

import asyncio
import tempfile
import pandas as pd
from pathlib import Path

from models import Action, ActionType, ExecutionContext
from code_generator import CodeGenerator
from sandbox import SandboxExecutor
from utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_sql_action_generation():
    """Test SQL action code generation and execution"""
    
    print("🔬 Testing SQL Action Generation and Execution")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = temp_dir
        
        # Create the CSV file first (like action 1 & 2 do)
        print("1️⃣ Creating test CSV file...")
        test_data = {
            'Rank': [1, 2, 3, 4, 5],
            'Peak': [1, 1, 3, 1, 5],
            'Title': ['Avatar', 'Avengers: Endgame', 'Avatar: The Way of Water', 'Titanic', 'Star Wars'],
            'Worldwide gross': ['$2,923,706,026', '$2,797,501,328', '$2,320,250,281', '$2,257,844,554', '$2,204,060,000'],
            'Year': [2009, 2019, 2022, 1997, 1977],
            'Ref': ['[# 1][# 2]', '[# 3][# 4]', '[# 5][# 6]', '[# 7][# 8]', '[# 9][# 10]']
        }
        
        df = pd.DataFrame(test_data)
        csv_path = Path(workspace_path) / "highest_grossing_films.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV created: {csv_path}")
        print(f"📊 Data shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Create the SQL action (action 3 from the plan)
        sql_action = Action(
            action_id="action_003",
            type=ActionType.SQL,
            description="Query for movies that grossed over $2 bn and were released before 2000",
            parameters={
                "query": "SELECT COUNT(*) FROM data WHERE Gross_Revenue >= 2000000000 AND Year < 2000"
            },
            output_files=["2bn_movies_before_2000.json"],
            dependencies=["action_002"]
        )
        
        print(f"\n2️⃣ Testing SQL Action...")
        print(f"🎯 Action: {sql_action.action_id}")
        print(f"📋 Query: {sql_action.parameters['query']}")
        
        # Create context
        context = ExecutionContext(
            workspace_path=workspace_path,
            available_files=["highest_grossing_films.csv"]
        )
        
        # Test code generation
        print(f"\n3️⃣ Generating code...")
        try:
            code_generator = CodeGenerator()
            generated_code = await code_generator.generate_code(sql_action, context)
            
            print("✅ Code generation successful")
            print(f"📝 Generated code length: {len(generated_code)} chars")
            print("📋 Generated code:")
            print("-" * 50)
            print(generated_code)
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Code generation failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Test sandbox execution
        print(f"\n4️⃣ Testing sandbox execution...")
        try:
            sandbox = SandboxExecutor()
            result = await sandbox.execute_code(generated_code, workspace_path, sql_action.action_id)
            
            if result["success"]:
                print("✅ Sandbox execution successful")
                print(f"📤 Output: {result.get('output', 'No output')}")
                print(f"📤 Stdout: {result.get('stdout', 'No stdout')}")
                
                # Check files created
                workspace = Path(workspace_path)
                json_files = list(workspace.glob("*.json"))
                all_files = list(workspace.glob("*"))
                
                print(f"📁 All files: {[f.name for f in all_files]}")
                print(f"📁 JSON files: {[f.name for f in json_files]}")
                
                if json_files:
                    # Read the result
                    import json
                    with open(json_files[0], 'r') as f:
                        result_data = json.load(f)
                    print(f"🎉 SQL result: {result_data}")
                
            else:
                print("❌ Sandbox execution failed")
                print(f"💥 Error: {result.get('error', 'Unknown error')}")
                print(f"📤 Stdout: {result.get('stdout', 'No stdout')}")
                print(f"📤 Stderr: {result.get('stderr', 'No stderr')}")
                
        except Exception as e:
            print(f"💥 Sandbox execution exception: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_sql_action_generation())
