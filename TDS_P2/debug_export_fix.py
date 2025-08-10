import asyncio
import tempfile
import os
from pathlib import Path

from code_generator import CodeGenerator
from models import Action, ExecutionContext, ActionType

async def test_export_fix():
    """Test the export action with the fixed instructions"""
    
    print("🔧 Testing Export Action Fix...")
    
    # Create a temporary workspace with the CSV data
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = temp_dir
        
        # Create the CSV file that would be created by previous actions
        csv_content = """Name,Age,Salary,Department
John,25,50000,Engineering
Alice,30,65000,Marketing
Bob,35,75000,Engineering
Carol,28,55000,Sales
David,32,70000,Engineering
Emma,29,60000,Marketing
Frank,40,80000,Sales
Grace,27,52000,Engineering"""
        
        csv_file = Path(workspace_path) / "dataframe.csv"
        with open(csv_file, 'w') as f:
            f.write(csv_content)
        
        print(f"✅ Created CSV file: {csv_file}")
        
        # Create export action
        export_action = Action(
            action_id="action_004",
            type=ActionType.EXPORT,
            description="Export final results as JSON array",
            parameters={
                "format": "json_array",
                "questions": [
                    "How many people work in the Engineering department?",
                    "What is the average salary across all departments?",
                    "Which department has the highest average salary?"
                ]
            },
            output_files=["final_results.json"]
        )
        
        # Create execution context
        context = ExecutionContext(
            workspace_path=workspace_path,
            available_files=["dataframe.csv"]
        )
        
        # Generate code
        code_generator = CodeGenerator()
        code = await code_generator.generate_code(export_action, context)
        
        print("\n📝 Generated Code:")
        print("-" * 50)
        print(code)
        print("-" * 50)
        
        # Execute the code
        from sandbox import SandboxExecutor
        sandbox = SandboxExecutor()
        
        result = await sandbox.execute_code(code, workspace_path, "test_export")
        
        print(f"\n🔍 Execution Result:")
        print(f"Success: {result['success']}")
        print(f"Stdout: {result.get('stdout', '')}")
        print(f"Stderr: {result.get('stderr', '')}")
        
        if result['success']:
            # Check if results file was created
            results_file = Path(workspace_path) / "final_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    content = f.read()
                print(f"\n📄 Results File Content: {content}")
                
                # Validate results
                import json
                try:
                    results = json.loads(content)
                    expected = ["4", "62125", "Engineering"]
                    print(f"\n✅ Expected: {expected}")
                    print(f"📋 Actual: {results}")
                    
                    if results == expected:
                        print("🎉 SUCCESS: Export action working correctly!")
                    else:
                        print("❌ FAILED: Results don't match expected values")
                except:
                    print("❌ FAILED: Could not parse results as JSON")
            else:
                print("❌ FAILED: Results file not created")
        else:
            print("❌ FAILED: Code execution failed")

if __name__ == "__main__":
    asyncio.run(test_export_fix())
