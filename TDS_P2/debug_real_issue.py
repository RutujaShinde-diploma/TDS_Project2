#!/usr/bin/env python3
"""
Debug the real issue by testing each action step by step
"""

import asyncio
import tempfile
from pathlib import Path

from models import JobRequest, Action, ActionType, ExecutionContext
from planner import PlannerModule
from code_generator import CodeGenerator
from sandbox import SandboxExecutor

async def debug_step_by_step():
    """Debug each step to find where it fails"""
    
    print("🐛 Step-by-Step Debugging")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = temp_dir
        
        # Create job request
        questions_text = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?"""
        
        job_request = JobRequest(questions=questions_text, files=[])
        
        print(f"📂 Workspace: {workspace_path}")
        
        # Step 1: Generate plan
        print("\n1️⃣ Generating plan...")
        planner = PlannerModule()
        plan = await planner.create_plan(job_request, workspace_path)
        print(f"✅ Plan has {len(plan.actions)} actions")
        
        # Test each action individually
        code_generator = CodeGenerator()
        sandbox = SandboxExecutor()
        
        for i, action in enumerate(plan.actions):
            print(f"\n{i+1}️⃣ Testing Action {action.action_id} ({action.type.value})")
            print(f"📋 Description: {action.description}")
            
            # Get current workspace files
            workspace = Path(workspace_path)
            available_files = [f.name for f in workspace.glob("*") if f.is_file()]
            
            context = ExecutionContext(
                workspace_path=workspace_path,
                available_files=available_files
            )
            
            try:
                # Generate code
                print(f"   🔧 Generating code...")
                code = await code_generator.generate_code(action, context)
                print(f"   ✅ Code generated ({len(code)} chars)")
                
                # Show first few lines of code
                code_lines = code.split('\n')[:5]
                print(f"   📝 Code preview: {' | '.join(code_lines)}")
                
                # Execute code
                print(f"   🏃 Executing in sandbox...")
                result = await sandbox.execute_code(code, workspace_path, action.action_id)
                
                if result["success"]:
                    print(f"   ✅ Execution successful")
                    stdout = result.get('stdout', '')[:100]
                    print(f"   📤 Output: {stdout}...")
                    
                    # Check files created
                    new_files = [f.name for f in workspace.glob("*") if f.is_file()]
                    created_files = [f for f in new_files if f not in available_files]
                    
                    if created_files:
                        print(f"   📁 Files created: {created_files}")
                    else:
                        print(f"   ⚠️ No new files created")
                        
                    # Check expected output files
                    for expected_file in action.output_files:
                        if expected_file in new_files:
                            print(f"   ✅ Expected file {expected_file} created")
                        else:
                            print(f"   ❌ Expected file {expected_file} NOT created")
                    
                else:
                    print(f"   ❌ Execution failed")
                    print(f"   💥 Error: {result.get('error', 'Unknown')}")
                    print(f"   📤 Stdout: {result.get('stdout', 'None')}")
                    print(f"   📤 Stderr: {result.get('stderr', 'None')}")
                    
                    # Show the full generated code for debugging
                    print(f"   📋 Full generated code:")
                    print("   " + "-" * 40)
                    for line in code.split('\n'):
                        print(f"   {line}")
                    print("   " + "-" * 40)
                    
                    break  # Stop on first failure
                    
            except Exception as e:
                print(f"   💥 Exception: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"\n📊 Final workspace contents:")
        final_files = [f.name for f in workspace.glob("*") if f.is_file()]
        print(f"Files: {final_files}")

if __name__ == "__main__":
    asyncio.run(debug_step_by_step())
