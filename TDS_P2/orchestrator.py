import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json

from config import config
from models import ExecutionPlan, Action, ActionResult, ActionStatus, ExecutionContext
from sandbox import SandboxExecutor
from code_generator import CodeGenerator
from utils.simple_storage import simple_storage

logger = logging.getLogger(__name__)

class Orchestrator:
    """Orchestrates the execution of action plans"""
    
    def __init__(self):
        self.sandbox = SandboxExecutor()
        self.code_generator = CodeGenerator()
        # Using simple storage instead of Redis cache
        # Using simple storage instead of Redis cache
        
    async def execute_plan(self, plan: ExecutionPlan, workspace_path: str) -> Union[List[str], Dict[str, str]]:
        """Execute a complete execution plan"""
        logger.info(f"🚀 ORCHESTRATOR: Starting execute_plan method")
        logger.info(f"🚀 ORCHESTRATOR: Plan ID: {plan.plan_id}")
        logger.info(f"🚀 ORCHESTRATOR: Workspace path: {workspace_path}")
        logger.info(f"🚀 ORCHESTRATOR: Number of actions: {len(plan.actions)}")
        logger.info(f"🚀 ORCHESTRATOR: Action types: {[action.type.value for action in plan.actions]}")
        
        start_time = time.time()
        results = []
        context = ExecutionContext(
            workspace_path=workspace_path,
            available_files=self._get_workspace_files(workspace_path)
        )
        
        logger.info(f"🚀 ORCHESTRATOR: Context created with {len(context.available_files)} available files")
        logger.info(f"🚀 ORCHESTRATOR: Available files: {context.available_files}")
        
        logger.info(f"Starting execution of plan {plan.plan_id} with {len(plan.actions)} actions")
        
        try:
            logger.info(f"🚀 ORCHESTRATOR: About to start action execution loop")
            # Execute actions sequentially
            for i, action in enumerate(plan.actions):
                logger.info(f"🚀 ORCHESTRATOR: Starting action {i+1}/{len(plan.actions)}: {action.action_id}")
                logger.info(f"Executing action {i+1}/{len(plan.actions)}: {action.action_id}")
                
                action_result = await self._execute_action(action, context)
                results.append(action_result)
                
                # Update context
                context.previous_results.append(action_result)
                context.available_files = self._get_workspace_files(workspace_path)
                
                # If action failed and retries exhausted, handle failure
                if action_result.status == ActionStatus.FAILED:
                    logger.error(f"Action {action.action_id} failed: {action_result.error}")
                    logger.error(f"Action output: {action_result.output}")
                    
                    # Try to continue with partial results if possible
                    if self._can_continue_with_failure(action, plan.actions[i+1:]):
                        logger.info("Continuing execution with partial results")
                        continue
                    else:
                        logger.error("Critical action failed, stopping execution")
                        break
                
                else:
                    # Log successful action for debugging
                    logger.info(f"Action {action.action_id} completed successfully")
                    if action_result.output and action_result.output.get("stdout"):
                        logger.debug(f"Action {action.action_id} output: {action_result.output['stdout'][:200]}...")
                    
                    # Verify and log file creation for debugging
                    workspace = Path(workspace_path)
                    created_files = list(workspace.glob("*"))
                    logger.info(f"Files in workspace after {action.action_id}: {[f.name for f in created_files]}")
                    
                    # Ensure output files exist
                    for expected_file in action.output_files:
                        file_path = workspace / expected_file
                        if file_path.exists():
                            logger.info(f"✅ Output file {expected_file} created successfully")
                        else:
                            logger.warning(f"⚠️ Expected output file {expected_file} not found")
                
                # Check time limit
                elapsed_time = time.time() - start_time
                if elapsed_time > config.MAX_EXECUTION_TIME:
                    logger.warning("Execution time limit reached")
                    break
            
            # Assemble final result
            final_result = await self._assemble_final_result(results, context, plan)
            
            execution_time = time.time() - start_time
            logger.info(f"Plan execution completed in {execution_time:.2f}s")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Plan execution failed: {str(e)}")
            raise Exception(f"Execution failed: {str(e)}")
    
    async def _execute_action(self, action: Action, context: ExecutionContext) -> ActionResult:
        """Execute a single action with retries"""
        logger.info(f"🚀 ORCHESTRATOR: Starting execution of {action.action_id}")
        logger.info(f"🚀 ORCHESTRATOR: Action type: {action.type.value}")
        logger.info(f"🚀 ORCHESTRATOR: Action description: {action.description}")
        logger.info(f"🚀 ORCHESTRATOR: Context workspace: {context.workspace_path}")
        logger.info(f"🚀 ORCHESTRATOR: Context available files: {context.available_files}")
        
        action_result = ActionResult(
            action_id=action.action_id,
            status=ActionStatus.IN_PROGRESS
        )
        
        # Special handling for LLM analysis actions
        if action.type.value == "llm_analysis":
            logger.info(f"Executing LLM analysis action {action.action_id}")
            try:
                # For LLM analysis, we don't need to execute code in sandbox
                # The analysis is done directly by the LLM during plan generation
                analysis_result = await self._execute_llm_analysis(action, context)
                
                action_result.status = ActionStatus.COMPLETED
                action_result.output = {
                    "success": True,
                    "result": analysis_result,
                    "stdout": str(analysis_result),
                    "stderr": "",
                    "return_code": 0
                }
                action_result.execution_time = 0.1  # Very fast for LLM analysis
                
                logger.info(f"LLM analysis action {action.action_id} completed successfully")
                return action_result
                
            except Exception as e:
                logger.error(f"LLM analysis action {action.action_id} failed: {str(e)}")
                action_result.status = ActionStatus.FAILED
                action_result.error = str(e)
                return action_result
        
        for attempt in range(config.MAX_RETRIES):
            try:
                logger.info(f"Executing {action.action_id}, attempt {attempt + 1}")
                
                # Generate or retrieve cached code
                code = await self._get_action_code(action, context)
                action_result.generated_code = code
                
                # Execute code in sandbox
                logger.info(f"🚀 ORCHESTRATOR: About to execute {action.action_id} in sandbox")
                logger.info(f"🚀 ORCHESTRATOR: Code length: {len(code)} characters")
                logger.info(f"🚀 ORCHESTRATOR: Workspace path: {context.workspace_path}")
                logger.info(f"🚀 ORCHESTRATOR: Available files: {context.available_files}")
                
                execution_result = await self.sandbox.execute_code(
                    code, context.workspace_path, action.action_id
                )
                
                if execution_result["success"]:
                    # Action succeeded
                    action_result.status = ActionStatus.COMPLETED
                    action_result.output = execution_result
                    action_result.execution_time = execution_result.get("execution_time")
                    
                    # Cache successful code
                    await self._cache_successful_code(action, context, code)
                    
                    logger.info(f"Action {action.action_id} completed successfully")
                    break
                
                else:
                    # Action failed, try to repair
                    error_msg = execution_result.get("error", "Unknown error")
                    stdout = execution_result.get("stdout", "")
                    stderr = execution_result.get("stderr", "")
                    return_code = execution_result.get("return_code", -1)
                    
                    logger.warning(f"Action {action.action_id} failed:")
                    logger.warning(f"  Error: {error_msg}")
                    logger.warning(f"  Return code: {return_code}")
                    logger.warning(f"  Stdout: {stdout}")
                    logger.warning(f"  Stderr: {stderr}")
                    
                    # Use more detailed error message
                    detailed_error = f"Return code: {return_code}, Error: {error_msg}"
                    if stderr:
                        detailed_error += f", Stderr: {stderr}"
                    if stdout:
                        detailed_error += f", Stdout: {stdout}"
                    
                    if attempt < config.MAX_RETRIES - 1:
                        # Try to repair the code
                        repaired_code = await self._repair_code(action, context, code, error_msg)
                        if repaired_code and repaired_code != code:
                            logger.info(f"Generated repair for {action.action_id}")
                            # Update code for next attempt (will be used in next iteration)
                            continue
                    
                    # If we reach here, either no more retries or repair failed
                    action_result.status = ActionStatus.FAILED
                    action_result.error = detailed_error
                    action_result.retry_count = attempt + 1
                    
            except Exception as e:
                logger.error(f"Error executing action {action.action_id}: {str(e)}")
                action_result.status = ActionStatus.FAILED
                action_result.error = str(e)
                action_result.retry_count = attempt + 1
        
        return action_result
    
    async def _get_action_code(self, action: Action, context: ExecutionContext) -> str:
        """Get code for action (from cache or generate new)"""
        # Try cache first
        cache_context = self._create_cache_context(action, context)
        cached_code = await self.code_cache.get_code(action.type.value, cache_context)
        
        if cached_code:
            logger.info(f"Using cached code for {action.action_id}")
            return cached_code
        
        # Generate new code
        logger.info(f"🚀 ORCHESTRATOR: Generating new code for {action.action_id}")
        code = await self.code_generator.generate_code(action, context)
        logger.info(f"🚀 ORCHESTRATOR: Generated code for {action.action_id}: {code[:200]}...")
        return code
    
    async def _repair_code(self, action: Action, context: ExecutionContext, 
                          failed_code: str, error_msg: str) -> Optional[str]:
        """Attempt to repair failed code"""
        try:
            repaired_code = await self.code_generator.repair_code(
                action, context, failed_code, error_msg
            )
            return repaired_code
        except Exception as e:
            logger.error(f"Code repair failed for {action.action_id}: {str(e)}")
            return None
    
    async def _execute_llm_analysis(self, action: Action, context: ExecutionContext) -> Dict[str, Any]:
        """Execute LLM analysis for embedded data"""
        try:
            # Read the questions file to get the embedded data
            questions_file = Path(context.workspace_path) / "questions.txt"
            if not questions_file.exists():
                raise Exception("Questions file not found")
            
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions_content = f.read()
            
            # Extract questions from the content
            lines = questions_content.strip().split('\n')
            questions = []
            for line in lines:
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    questions.append(line.strip())
            
            # Use OpenAI to analyze the embedded data
            import openai
            client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            
            system_prompt = """You are an expert data analyst. Analyze the embedded data and answer the questions accurately.

For CSV data, calculate exact values:
- Count rows, calculate averages, sums, etc.
- Identify patterns and relationships
- Provide precise numerical answers

IMPORTANT: Answer each question clearly and concisely. Do not repeat information. Format your response as:
1. [Question 1] - [Clear answer with calculation]
2. [Question 2] - [Clear answer with calculation]
3. [Question 3] - [Clear answer with calculation]

Keep each answer focused and avoid redundancy."""
            
            user_prompt = f"""Analyze this data and answer the questions:

{questions_content}

Please provide clear, concise answers to each question with calculations. Do not repeat information."""
            
            response = await client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            analysis_result = response.choices[0].message.content
            
            return {
                "analysis": analysis_result,
                "questions_answered": len(questions),
                "data_type": "embedded_csv"
            }
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            raise Exception(f"LLM analysis failed: {str(e)}")
    
    async def _cache_successful_code(self, action: Action, context: ExecutionContext, code: str):
        """Cache successful code for future use"""
        try:
            cache_context = self._create_cache_context(action, context)
            await self.code_cache.cache_code(action.type.value, cache_context, code)
        except Exception as e:
            logger.warning(f"Failed to cache code: {str(e)}")
    
    def _create_cache_context(self, action: Action, context: ExecutionContext) -> str:
        """Create a context string for caching"""
        context_parts = [
            action.type.value,
            str(sorted(action.parameters.items())),
            str(sorted(context.available_files))
        ]
        
        # For export actions, include the questions content to avoid cache conflicts
        if action.type.value == "export" and "questions" in action.parameters:
            questions_content = action.parameters.get("questions", "")
            # Use a hash of the questions to keep cache key manageable
            import hashlib
            questions_hash = hashlib.md5(questions_content.encode()).hexdigest()[:8]
            context_parts.append(f"questions_hash:{questions_hash}")
        
        return "|".join(context_parts)
    
    def _can_continue_with_failure(self, failed_action: Action, remaining_actions: List[Action]) -> bool:
        """Determine if execution can continue after an action failure"""
        # Check if any remaining actions depend on the failed action
        failed_outputs = failed_action.output_files + failed_action.output_variables
        
        for action in remaining_actions:
            action_inputs = action.input_files + action.input_variables + action.dependencies
            
            # If any remaining action needs output from failed action, we can't continue
            if any(output in action_inputs for output in failed_outputs):
                return False
            
            # If this is an export action (final step), we can't continue
            if action.type.value == "export":
                return False
        
        return True
    
    async def _assemble_final_result(self, results: List[ActionResult], 
                                   context: ExecutionContext, 
                                   plan: ExecutionPlan) -> Union[List[str], Dict[str, str]]:
        """Assemble the final result from all action results"""
        try:
            workspace = Path(context.workspace_path)
            
            # Strategy 1: Look for final output files
            output_files = ["final_results.json", "final_response.json", "results.json", "query_result.json"]
            for filename in output_files:
                file_path = workspace / filename
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            final_result = json.load(f)
                        logger.info(f"Found final results in {filename}")
                        # If it's already a dictionary with answer keys, return it directly
                        if isinstance(final_result, dict) and any(key.startswith('answer') for key in final_result.keys()):
                            return final_result
                        # If it's a list, return it directly
                        elif isinstance(final_result, list):
                            return final_result
                        # Otherwise, convert to string and wrap in list
                        else:
                            return [str(final_result)]
                    except Exception as e:
                        logger.warning(f"Failed to parse {filename}: {e}")
            
            # Strategy 2: Find export action and check its output files
            for action in plan.actions:
                if action.type.value == "export":
                    for output_file in action.output_files:
                        file_path = workspace / output_file
                        if file_path.exists():
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read().strip()
                                    # Try to parse as JSON first
                                    try:
                                        result = json.loads(content)
                                        return result if isinstance(result, list) else [str(result)]
                                    except:
                                        # Return as string if not JSON
                                        return [content]
                            except Exception as e:
                                logger.warning(f"Failed to read {output_file}: {e}")
            
            # Strategy 3: Compile results from successful actions' stdout
            compiled_results = []
            for i, result in enumerate(results):
                if result.status == ActionStatus.COMPLETED and result.output:
                    # Check if this is an LLM analysis result
                    if result.output.get("result") and isinstance(result.output["result"], dict):
                        llm_result = result.output["result"]
                        if "analysis" in llm_result:
                            # Format LLM analysis results nicely
                            analysis_text = llm_result["analysis"]
                            # Clean up the analysis text
                            clean_analysis = self._clean_llm_analysis(analysis_text)
                            compiled_results.append(clean_analysis)
                            continue
                    
                    stdout = result.output.get("stdout", "")
                    if stdout and stdout.strip():
                        # Clean up the output
                        clean_output = stdout.strip()
                        # Skip generic messages and data loading info
                        if not any(phrase in clean_output.lower() for phrase in 
                                 ["starting", "fetching", "found", "saved", "completed", "loaded", "rows", "columns", "dataframe", "rangeindex"]):
                            compiled_results.append(clean_output)
            
            # Strategy 4: Look for any JSON files with answers
            for json_file in workspace.glob("*.json"):
                if json_file.name not in ["plan.json", "metadata.json"]:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            content = json.load(f)
                        if content:  # Non-empty content
                            compiled_results.append(f"From {json_file.name}: {content}")
                    except:
                        continue
            
            if compiled_results:
                # Try to parse into simple numbered answers
                simple_answers = self._parse_to_simple_answers(compiled_results)
                if simple_answers:
                    return simple_answers
                return compiled_results
            
            # Strategy 5: Check if scraping worked by looking for CSV files
            csv_files = list(workspace.glob("*.csv"))
            if csv_files:
                return [f"Data was scraped successfully. Found {len(csv_files)} data files, but analysis results were not properly formatted."]
            
            return ["Analysis completed with partial results"]
            
        except Exception as e:
            logger.error(f"Error assembling final result: {str(e)}")
            return [f"Analysis completed with errors: {str(e)}"]
    
    def _get_workspace_files(self, workspace_path: str) -> List[str]:
        """Get list of files in workspace"""
        try:
            workspace = Path(workspace_path)
            files = []
            for item in workspace.iterdir():
                if item.is_file():
                    files.append(item.name)
            return files
        except Exception:
            return []
    
    def _clean_llm_analysis(self, analysis_text: str) -> str:
        """Clean and format LLM analysis results"""
        try:
            # Remove duplicate sections and clean up formatting
            lines = analysis_text.strip().split('\n')
            cleaned_lines = []
            seen_sections = set()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is a new question section
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    section_key = line[:3]  # e.g., "1.", "2.", etc.
                    if section_key not in seen_sections:
                        seen_sections.add(section_key)
                        cleaned_lines.append("")  # Add spacing
                        cleaned_lines.append(line)
                    else:
                        # Skip duplicate sections
                        continue
                else:
                    # Add non-question lines
                    cleaned_lines.append(line)
            
            # Join lines and clean up extra whitespace
            cleaned_text = "\n".join(cleaned_lines)
            
            # Remove extra blank lines
            import re
            cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
            
            return cleaned_text.strip()
            
        except Exception as e:
            logger.warning(f"Failed to clean LLM analysis: {e}")
            return analysis_text  # Return original if cleaning fails
    
    def _parse_to_simple_answers(self, results: List[str]) -> Optional[Dict[str, str]]:
        """Parse raw results into simple numbered answers"""
        try:
            answers = {}
            answer_count = 1
            
            # First, try to find JSON files with structured answers
            for result in results:
                if "From " in result and ".json" in result:
                    try:
                        # Extract the JSON content from the result
                        json_start = result.find("{")
                        if json_start != -1:
                            json_content = result[json_start:]
                            import json
                            parsed_json = json.loads(json_content)
                            if isinstance(parsed_json, dict):
                                # If it's already in answer1, answer2 format, return it
                                if any(key.startswith('answer') for key in parsed_json.keys()):
                                    return parsed_json
                                # Otherwise, convert to answer format
                                for key, value in parsed_json.items():
                                    answers[f"answer{answer_count}"] = str(value)
                                    answer_count += 1
                                return answers
                    except:
                        continue
            
            # Extract answers from text results
            for result in results:
                if "Total sales:" in result:
                    value = result.split(":")[1].strip()
                    answers[f"answer{answer_count}"] = value
                    answer_count += 1
                elif "Region with highest" in result:
                    value = result.split(":")[1].strip()
                    answers[f"answer{answer_count}"] = value
                    answer_count += 1
                elif "Correlation between day" in result:
                    value = result.split(":")[1].strip()
                    answers[f"answer{answer_count}"] = value
                    answer_count += 1
                elif "Median sales:" in result:
                    value = result.split(":")[1].strip()
                    answers[f"answer{answer_count}"] = value
                    answer_count += 1
                elif "Total salary:" in result or "Total salaries:" in result:
                    value = result.split(":")[1].strip()
                    answers[f"answer{answer_count}"] = value
                    answer_count += 1
                elif "Average age:" in result or "Mean age:" in result:
                    value = result.split(":")[1].strip()
                    answers[f"answer{answer_count}"] = value
                    answer_count += 1
                elif "Highest salary:" in result or "Max salary:" in result:
                    value = result.split(":")[1].strip()
                    answers[f"answer{answer_count}"] = value
                    answer_count += 1
                elif "base64" in result.lower() or "data:image" in result:
                    # Extract base64 data
                    if "data:image/png;base64," in result:
                        base64_data = result.split("data:image/png;base64,")[1]
                        answers[f"answer{answer_count}"] = base64_data
                        answer_count += 1
            
            return answers if answers else None
            
        except Exception as e:
            logger.error(f"Error parsing results to simple answers: {str(e)}")
            return None
