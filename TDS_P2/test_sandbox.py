#!/usr/bin/env python3
"""
Test script to verify sandbox execution works
"""

import asyncio
import sys
from pathlib import Path

async def test_sandbox():
    """Test the sandbox execution"""
    print("🧪 Testing sandbox execution...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from sandbox import SandboxExecutor
        print("   ✅ Sandbox import successful")
        
        # Test sandbox
        print("2. Testing sandbox...")
        sandbox = SandboxExecutor()
        
        # Create test workspace
        workspace_path = "test_sandbox_workspace"
        Path(workspace_path).mkdir(exist_ok=True)
        
        # Test simple Python code
        test_code = """
import pandas as pd
import json

# Create a simple DataFrame
df = pd.DataFrame({'sales': [100, 200, 150, 50, 120, 220, 130, 170]})
total_sales = df['sales'].sum()

# Save result
result = {"total_sales": int(total_sales)}
with open('test_result.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"Total sales: {total_sales}")
print("Result saved to test_result.json")
"""
        
        print("   🔍 Test code:")
        print(test_code)
        
        # Execute code
        print("3. Executing test code...")
        result = await sandbox.execute_code(test_code, workspace_path, "test_action")
        
        print(f"   📊 Execution result: {result}")
        
        # Check if output file was created
        output_file = Path(workspace_path) / "test_result.json"
        if output_file.exists():
            print(f"   ✅ Output file created: {output_file}")
            with open(output_file, 'r') as f:
                content = f.read()
                print(f"   📄 File content: {content}")
        else:
            print(f"   ❌ Output file not found: {output_file}")
        
        # Check workspace contents
        workspace_files = list(Path(workspace_path).glob("*"))
        print(f"   📁 Workspace files: {[f.name for f in workspace_files]}")
        
        # Cleanup
        import shutil
        shutil.rmtree(workspace_path, ignore_errors=True)
        
        print("\n🎉 Sandbox test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_sandbox())
    sys.exit(0 if success else 1)
