#!/usr/bin/env python3
"""
Test if orchestrator can be instantiated and basic methods work
"""

try:
    from orchestrator import Orchestrator
    print("✅ Orchestrator import successful")
    
    orchestrator = Orchestrator()
    print("✅ Orchestrator instantiation successful")
    
    # Test if _get_workspace_files works
    test_path = "."
    files = orchestrator._get_workspace_files(test_path)
    print(f"✅ _get_workspace_files works: found {len(files)} files")
    
except Exception as e:
    print(f"❌ Orchestrator error: {e}")
    import traceback
    traceback.print_exc()
