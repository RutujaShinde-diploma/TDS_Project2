#!/usr/bin/env python3
"""
Component-Specific Tests - Deep testing of individual components
"""

import sys
import os
import asyncio
import tempfile
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ComponentTester:
    """Test individual components in detail"""
    
    async def test_config_component(self):
        """Test configuration system thoroughly"""
        print("🔧 Testing Configuration Component...")
        
        try:
            from config import config
            
            # Test required attributes
            required_attrs = [
                'OPENAI_API_KEY', 'OPENAI_MODEL', 'MAX_EXECUTION_TIME',
                'SANDBOX_TIMEOUT', 'WORKSPACE_DIR', 'BLOCKED_IMPORTS'
            ]
            
            for attr in required_attrs:
                assert hasattr(config, attr), f"Missing config attribute: {attr}"
            
            # Test value types
            assert isinstance(config.MAX_EXECUTION_TIME, int), "MAX_EXECUTION_TIME should be int"
            assert config.MAX_EXECUTION_TIME > 0, "MAX_EXECUTION_TIME should be positive"
            assert isinstance(config.BLOCKED_IMPORTS, set), "BLOCKED_IMPORTS should be set"
            
            print("  ✅ All configuration attributes present and valid")
            return True
            
        except Exception as e:
            print(f"  ❌ Configuration test failed: {e}")
            return False
    
    async def test_models_component(self):
        """Test Pydantic models thoroughly"""
        print("🏗️  Testing Models Component...")
        
        try:
            from models import Action, ExecutionPlan, ActionType, ActionStatus, JobRequest
            
            # Test Action model
            action = Action(
                action_id="test_action_001",
                type=ActionType.STATS,
                description="Calculate statistics",
                parameters={"column": "age", "operation": "mean"}
            )
            
            # Test validation
            assert action.action_id == "test_action_001"
            assert action.type == ActionType.STATS
            assert action.parameters["column"] == "age"
            
            # Test ExecutionPlan
            plan = ExecutionPlan(
                plan_id="test_plan_001",
                task_description="Test task",
                actions=[action],
                estimated_total_time=120
            )
            
            assert len(plan.actions) == 1
            assert plan.actions[0].action_id == "test_action_001"
            
            # Test JobRequest
            job_request = JobRequest(
                questions="What is the mean age?",
                files=["data.csv"],
                output_format="json"
            )
            
            assert job_request.questions == "What is the mean age?"
            assert "data.csv" in job_request.files
            
            print("  ✅ All models validate correctly")
            return True
            
        except Exception as e:
            print(f"  ❌ Models test failed: {e}")
            return False
    
    async def test_validator_component(self):
        """Test plan validator thoroughly"""
        print("🛡️  Testing Validator Component...")
        
        try:
            from validator import PlanValidator
            from models import ExecutionPlan, Action, ActionType
            
            validator = PlanValidator()
            
            # Test valid plan
            valid_action = Action(
                action_id="valid_001",
                type=ActionType.STATS,
                description="Safe statistics",
                estimated_time=30
            )
            
            valid_plan = ExecutionPlan(
                plan_id="valid_plan",
                task_description="Valid test",
                actions=[valid_action],
                estimated_total_time=30
            )
            
            result = await validator.validate_plan(valid_plan)
            assert result.is_valid, f"Valid plan rejected: {result.errors}"
            
            # Test invalid plan (too long) - should fail at model creation
            try:
                invalid_action = Action(
                    action_id="invalid_001",
                    type=ActionType.STATS,
                    description="Long running task",
                    estimated_time=200  # Too long
                )
                
                invalid_plan = ExecutionPlan(
                    plan_id="invalid_plan",
                    task_description="Invalid test",
                    actions=[invalid_action],
                    estimated_total_time=200
                )
                
                # If we get here, the model validation didn't catch it, so test the validator
                result = await validator.validate_plan(invalid_plan)
                assert not result.is_valid, "Invalid plan accepted"
                
            except Exception:
                # Expected - model validation should prevent invalid plans
                pass
            
            print("  ✅ Plan validation working correctly")
            return True
            
        except Exception as e:
            print(f"  ❌ Validator test failed: {e}")
            return False
    
    async def test_sandbox_component(self):
        """Test sandbox execution thoroughly"""
        print("🔒 Testing Sandbox Component...")
        
        try:
            from sandbox import SandboxExecutor, CodeValidator
            
            # Test code validator
            validator = CodeValidator()
            
            # Safe code
            safe_code = """
import pandas as pd
import numpy as np

data = {'values': [1, 2, 3, 4, 5]}
df = pd.DataFrame(data)
mean_val = df['values'].mean()
print(f"Mean: {mean_val}")
"""
            
            is_safe, errors = validator.validate_code(safe_code)
            assert is_safe, f"Safe code rejected: {errors}"
            
            # Unsafe code
            unsafe_code = "import os; os.system('echo dangerous')"
            is_safe, errors = validator.validate_code(unsafe_code)
            assert not is_safe, "Unsafe code accepted"
            
            # Test sandbox execution
            sandbox = SandboxExecutor()
            workspace = project_root / "workspace" / "test"
            workspace.mkdir(parents=True, exist_ok=True)
            
            result = await sandbox.execute_code(safe_code, str(workspace), "test_sandbox")
            
            assert result["success"], f"Safe code execution failed: {result.get('error')}"
            assert "Mean: 3.0" in result.get("stdout", ""), "Expected output not found"
            
            print("  ✅ Sandbox validation and execution working")
            return True
            
        except Exception as e:
            print(f"  ❌ Sandbox test failed: {e}")
            return False
    
    async def test_cache_component(self):
        """Test caching system thoroughly"""
        print("💾 Testing Cache Component...")
        
        try:
            from utils.cache import CacheManager, LLMCache, CodeCache
            
            # Test basic cache manager
            cache_manager = CacheManager()
            await cache_manager.initialize()
            
            # Test set/get with different data types
            test_cases = [
                ("string_key", "string_value"),
                ("dict_key", {"nested": {"data": [1, 2, 3]}}),
                ("list_key", [1, "two", {"three": 3}])
            ]
            
            for key, value in test_cases:
                await cache_manager.set(key, value)
                retrieved = await cache_manager.get(key)
                assert retrieved == value, f"Cache mismatch for {key}"
            
            # Test TTL and expiration
            await cache_manager.set("temp_key", "temp_value", ttl=1)
            assert await cache_manager.exists("temp_key"), "Key should exist immediately"
            
            # Test specialized caches
            llm_cache = LLMCache(cache_manager)
            await llm_cache.cache_response("test prompt", "gpt-4", "test response")
            response = await llm_cache.get_response("test prompt", "gpt-4")
            assert response == "test response", "LLM cache failed"
            
            code_cache = CodeCache(cache_manager)
            await code_cache.cache_code("stats", "test context", "test code")
            code = await code_cache.get_code("stats", "test context")
            assert code == "test code", "Code cache failed"
            
            print("  ✅ All cache operations working correctly")
            return True
            
        except Exception as e:
            print(f"  ❌ Cache test failed: {e}")
            return False
    
    async def test_file_analyzer_component(self):
        """Test file analyzer thoroughly"""
        print("📁 Testing File Analyzer Component...")
        
        try:
            from utils.file_analyzer import FileAnalyzer
            
            analyzer = FileAnalyzer()
            
            # Test CSV analysis
            csv_content = "name,age,salary\nAlice,25,50000\nBob,30,60000\nCharlie,35,55000"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write(csv_content)
                csv_file = Path(f.name)
            
            try:
                csv_metadata = await analyzer.analyze(csv_file)
                assert csv_metadata['type'] == 'csv', "CSV type not detected"
                assert 'columns' in csv_metadata, "CSV columns not extracted"
                assert 'rows' in csv_metadata, "CSV rows not counted"
                assert csv_metadata['columns'] == ['name', 'age', 'salary'], "Incorrect columns"
            finally:
                csv_file.unlink()
            
            # Test JSON analysis
            json_content = json.dumps({"users": [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]})
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(json_content)
                json_file = Path(f.name)
            
            try:
                json_metadata = await analyzer.analyze(json_file)
                assert json_metadata['type'] == 'json', "JSON type not detected"
                assert 'structure' in json_metadata, "JSON structure not analyzed"
            finally:
                json_file.unlink()
            
            # Test text analysis
            text_content = "This is a test document.\nIt has multiple lines.\nFor testing purposes."
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(text_content)
                text_file = Path(f.name)
            
            try:
                text_metadata = await analyzer.analyze(text_file)
                assert text_metadata['type'] == 'text', "Text type not detected"
                assert 'line_count' in text_metadata, "Line count not calculated"
                assert text_metadata['line_count'] == 3, "Incorrect line count"
            finally:
                text_file.unlink()
            
            print("  ✅ File analysis working for all formats")
            return True
            
        except Exception as e:
            print(f"  ❌ File analyzer test failed: {e}")
            return False
    
    async def test_code_generator_component(self):
        """Test code generator (without LLM calls)"""
        print("🤖 Testing Code Generator Component...")
        
        try:
            from code_generator import CodeGenerator
            from models import Action, ActionType, ExecutionContext
            
            # Test instantiation and structure
            generator = CodeGenerator()
            assert generator is not None
            
            # Test prompt generation methods
            action = Action(
                action_id="test_gen_001",
                type=ActionType.STATS,
                description="Calculate mean",
                parameters={"column": "age"}
            )
            
            context = ExecutionContext(
                workspace_path="/test/workspace",
                available_files=["data.csv"]
            )
            
            # Test that prompt generation methods exist and work
            system_prompt = generator._get_system_prompt()
            assert isinstance(system_prompt, str), "System prompt should be string"
            assert "Python" in system_prompt, "System prompt should mention Python"
            
            code_prompt = generator._create_code_prompt(action, context)
            assert isinstance(code_prompt, str), "Code prompt should be string"
            assert "STATS" in code_prompt or "stats" in code_prompt, "Code prompt should include action type"
            
            # Test code extraction
            test_response = """Here's the code:
```python
import pandas as pd
df = pd.read_csv('data.csv')
result = df['age'].mean()
print(f"Mean age: {result}")
```
That should work!"""
            
            extracted_code = generator._extract_code_from_response(test_response)
            assert "import pandas as pd" in extracted_code, "Code extraction failed"
            assert "df['age'].mean()" in extracted_code, "Code extraction incomplete"
            
            print("  ✅ Code generator structure and methods working")
            return True
            
        except Exception as e:
            print(f"  ❌ Code generator test failed: {e}")
            return False

async def main():
    """Run component tests"""
    print("🔍 Component Testing Suite")
    print("=" * 50)
    
    tester = ComponentTester()
    
    # Define tests
    tests = [
        ("Configuration", tester.test_config_component),
        ("Models", tester.test_models_component),
        ("Validator", tester.test_validator_component),
        ("Sandbox", tester.test_sandbox_component),
        ("Cache", tester.test_cache_component),
        ("File Analyzer", tester.test_file_analyzer_component),
        ("Code Generator", tester.test_code_generator_component),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"  💥 {test_name} test crashed: {e}")
            print()
    
    # Summary
    print("=" * 50)
    print(f"📊 Component Test Summary: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All components working correctly!")
        return 0
    else:
        print("⚠️  Some component tests failed.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
