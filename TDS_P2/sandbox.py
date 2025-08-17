import asyncio
import tempfile
import os
import shutil
import logging
import time
import ast
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

# Docker support removed for simplified deployment
DOCKER_AVAILABLE = False

from config import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class CodeValidator:
    """Static code analysis for safety checks"""
    
    def __init__(self):
        self.blocked_imports = config.BLOCKED_IMPORTS
        self.dangerous_functions = {
            'exec', 'eval', 'compile', '__import__', 'input',
            'raw_input', 'reload', 'vars', 'globals', 'locals'
            # Removed 'open', 'exit', 'quit' - legitimate for file operations and error handling
        }
    
    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """Validate code for security issues"""
        errors = []
        
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Check imports
            import_errors = self._check_imports(tree)
            errors.extend(import_errors)
            
            # Check function calls
            function_errors = self._check_function_calls(tree)
            errors.extend(function_errors)
            
            # Check string patterns
            pattern_errors = self._check_string_patterns(code)
            errors.extend(pattern_errors)
            
            is_safe = len(errors) == 0
            return is_safe, errors
            
        except SyntaxError as e:
            return False, [f"Syntax error: {str(e)}"]
        except Exception as e:
            return False, [f"Code validation error: {str(e)}"]
    
    def _check_imports(self, tree: ast.AST) -> List[str]:
        """Check for dangerous imports"""
        errors = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.blocked_imports:
                        errors.append(f"Blocked import: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module in self.blocked_imports:
                    errors.append(f"Blocked import: {node.module}")
        
        return errors
    
    def _check_function_calls(self, tree: ast.AST) -> List[str]:
        """Check for dangerous function calls"""
        errors = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.dangerous_functions:
                        errors.append(f"Dangerous function call: {node.func.id}")
        
        return errors
    
    def _check_string_patterns(self, code: str) -> List[str]:
        """Check for dangerous string patterns"""
        errors = []
        
        # Only block truly dangerous patterns, allow legitimate usage
        dangerous_patterns = [
            r'import\s+subprocess',
            r'__import__\s*\(',
            r'exec\s*\(',
            r'eval\s*\(',
            r'subprocess\.',
            # Allow sys imports and os for legitimate file operations
            # r'import\s+os',   # Removed - needed for file operations
            # r'import\s+sys',  # Removed - needed for error handling 
            # r'open\s*\(',     # Removed - needed for file operations
            # r'file\s*\(',     # Removed - legitimate usage
            # r'os\.',          # Removed - needed for file operations
            # r'sys\.'          # Removed - needed for error handling
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(f"Dangerous pattern detected: {pattern}")
        
        return errors

class SandboxExecutor:
    """Execute code in a secure sandbox environment"""
    
    def __init__(self):
        self.docker_client = None
        self.validator = CodeValidator()
    

    
    async def execute_code(self, code: str, workspace_path: str, action_id: str) -> Dict[str, Any]:
        """Execute code in sandbox"""
        start_time = time.time()
        
        try:
            # Validate code first
            is_safe, validation_errors = self.validator.validate_code(code)
            if not is_safe:
                return {
                    "success": False,
                    "error": f"Code validation failed: {'; '.join(validation_errors)}",
                    "execution_time": time.time() - start_time
                }
            
            # Using subprocess execution (simplified deployment)
            logger.info("Using subprocess execution")
            result = await self._execute_in_subprocess(code, workspace_path, action_id)
            
            result["execution_time"] = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Sandbox execution error: {str(e)}")
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "execution_time": time.time() - start_time
            }
    

    
    async def _execute_in_subprocess(self, code: str, workspace_path: str, action_id: str) -> Dict[str, Any]:
        """Execute code in subprocess (simplified deployment)"""
        try:
            # Create code file in workspace using absolute path
            workspace_path_abs = os.path.abspath(workspace_path)
            code_file = os.path.join(workspace_path_abs, f"{action_id}.py")
            
            logger.info(f"🔍 SANDBOX DEBUG: Workspace path: {workspace_path}")
            logger.info(f"🔍 SANDBOX DEBUG: Absolute workspace path: {workspace_path_abs}")
            logger.info(f"🔍 SANDBOX DEBUG: Code file path: {code_file}")
            logger.info(f"🔍 SANDBOX DEBUG: Code file exists: {os.path.exists(code_file)}")
            logger.info(f"🔍 SANDBOX DEBUG: Workspace contents: {os.listdir(workspace_path_abs) if os.path.exists(workspace_path_abs) else 'Workspace does not exist'}")
            
            wrapped_code = self._wrap_code(code)
            logger.info(f"🔍 SANDBOX DEBUG: Generated code length: {len(code)} characters")
            logger.info(f"🔍 SANDBOX DEBUG: Wrapped code length: {len(wrapped_code)} characters")
            logger.info(f"🔍 SANDBOX DEBUG: Code preview: {code[:200]}...")
            
            with open(code_file, 'w') as f:
                f.write(wrapped_code)
            
            logger.info(f"🔍 SANDBOX DEBUG: Code file written, size: {os.path.getsize(code_file)} bytes")
            logger.info(f"🔍 SANDBOX DEBUG: Python3 executable: {shutil.which('python3')}")
            logger.info(f"🔍 SANDBOX DEBUG: Environment PATH: {os.environ.get('PATH', 'Not set')}")
            
            restricted_env = self._get_restricted_env()
            logger.info(f"🔍 SANDBOX DEBUG: Restricted env PATH: {restricted_env.get('PATH', 'Not set')}")
            logger.info(f"🔍 SANDBOX DEBUG: Restricted env keys: {list(restricted_env.keys())}")
            
            # Execute with timeout using absolute path
            # Try to find the correct Python executable
            python_exec = shutil.which('python3') or shutil.which('python') or 'python3'
            logger.info(f"🔍 SANDBOX DEBUG: Using Python executable: {python_exec}")
            
            process = await asyncio.create_subprocess_exec(
                python_exec, code_file,
                cwd=workspace_path_abs,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=restricted_env
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.SANDBOX_TIMEOUT
                )
                
                stdout_str = stdout.decode() if stdout else ""
                stderr_str = stderr.decode() if stderr else ""
                
                success = process.returncode == 0
                if not success:
                    error_msg = f"Process failed with return code {process.returncode}"
                    if stderr_str:
                        error_msg += f": {stderr_str}"
                    elif stdout_str:
                        error_msg += f": {stdout_str}"
                    else:
                        error_msg += " (no error output)"
                else:
                    error_msg = ""
                
                return {
                    "success": success,
                    "output": stdout_str,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "return_code": process.returncode,
                    "error": error_msg if not success else ""
                }
                
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "success": False,
                    "error": f"Execution timeout ({config.SANDBOX_TIMEOUT}s)",
                    "stdout": "",
                    "stderr": ""
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Subprocess execution error: {str(e)}"
            }
        finally:
            # Cleanup code file
            try:
                if os.path.exists(code_file):
                    os.remove(code_file)
            except:
                pass
    
    def _wrap_code(self, code: str) -> str:
        """Wrap user code with safety measures"""
        wrapper = f"""
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

# Redirect to prevent blocking operations
class SafeStdout:
    def write(self, text):
        sys.__stdout__.write(str(text))
    def flush(self):
        sys.__stdout__.flush()

sys.stdout = SafeStdout()

try:
    # User code starts here
{self._indent_code(code)}
    # User code ends here
    
except Exception as e:
    print(f"EXECUTION_ERROR: {{type(e).__name__}}: {{str(e)}}")
    traceback.print_exc()
    sys.exit(1)
"""
        return wrapper
    
    def _indent_code(self, code: str) -> str:
        """Indent code for wrapping"""
        lines = code.split('\n')
        indented_lines = ['    ' + line for line in lines]
        return '\n'.join(indented_lines)
    
    def _get_restricted_env(self) -> Dict[str, str]:
        """Get restricted environment variables"""
        env = os.environ.copy()
        
        # Remove potentially dangerous env vars
        dangerous_vars = [
            'PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH',
            'HOME', 'USER', 'USERNAME'
        ]
        
        for var in dangerous_vars:
            env.pop(var, None)
        
        # Set minimal required vars
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONUNBUFFERED'] = '1'
        
        return env
