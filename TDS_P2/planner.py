import json
import openai
from typing import List, Dict, Any
import logging
from pathlib import Path
import hashlib

from config import config
from models import JobRequest, ExecutionPlan, Action, ActionType
from utils.simple_storage import simple_storage
from utils.file_analyzer import FileAnalyzer
from utils.logger import setup_logger

logger = setup_logger(__name__)

class PlannerModule:
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        # Using simple storage instead of Redis cache
        self.file_analyzer = FileAnalyzer()
        
    async def create_plan(self, job_request: JobRequest, workspace_path: str) -> ExecutionPlan:
        """Create an execution plan for the given job request"""
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(job_request, workspace_path)
            
            # Try to get from simple storage first
            cached_plan = simple_storage.get(f"plan:{cache_key}")
            if cached_plan:
                logger.info("Using cached execution plan")
                return ExecutionPlan.parse_obj(cached_plan)
            
            # Analyze uploaded files
            file_metadata = await self._analyze_files(job_request.files, workspace_path)
            
            # Generate plan using LLM
            plan_json = await self._generate_plan_with_llm(job_request, file_metadata)
            
            # Parse and validate the plan
            plan = self._parse_plan(plan_json, job_request.questions)
            
            # Cache the plan in simple storage
            simple_storage.set(f"plan:{cache_key}", plan.dict(), ttl=config.CACHE_TTL)
            
            logger.info(f"Generated execution plan with {len(plan.actions)} actions")
            return plan
            
        except Exception as e:
            logger.error(f"Error creating plan: {str(e)}")
            raise Exception(f"Failed to create execution plan: {str(e)}")
    
    def _generate_cache_key(self, job_request: JobRequest, workspace_path: str) -> str:
        """Generate a cache key for the job request including file timestamps"""
        # Include file modification times in cache key to detect changes
        file_timestamps = []
        for filename in job_request.files:
            file_path = Path(workspace_path) / filename
            if file_path.exists():
                try:
                    mtime = file_path.stat().st_mtime
                    file_timestamps.append(f"{filename}:{mtime}")
                except Exception:
                    file_timestamps.append(f"{filename}:unknown")
        
        # Create cache key with content and timestamps
        content = job_request.questions + "|" + "|".join(sorted(file_timestamps))
        cache_key = hashlib.md5(content.encode()).hexdigest()
        
        logger.info(f"Generated cache key: {cache_key[:8]}... for content: {job_request.questions[:50]}...")
        return cache_key
    
    async def _analyze_files(self, files: List[str], workspace_path: str) -> Dict[str, Any]:
        """Analyze uploaded files to understand their structure and content"""
        file_metadata = {}
        
        for filename in files:
            file_path = Path(workspace_path) / filename
            if file_path.exists():
                metadata = await self.file_analyzer.analyze(file_path)
                file_metadata[filename] = metadata
        
        return file_metadata
    
    def _analyze_task_requirements(self, questions_text: str) -> Dict[str, Any]:
        """Analyze the questions text to understand task requirements"""
        task_analysis = {
            "task_type": "unknown",
            "requires_external_files": False,
            "requires_visualization": False,
            "requires_network_analysis": False,
            "requires_web_scraping": False,
            "has_embedded_data": False,
            "data_preview": "",
            "referenced_files": []
        }
        
        lines = questions_text.strip().split('\n')
        text_lower = questions_text.lower()
        
        # Check for external file references
        if 'edges.csv' in text_lower:
            task_analysis["requires_external_files"] = True
            task_analysis["requires_network_analysis"] = True
            task_analysis["referenced_files"].append("edges.csv")
            task_analysis["task_type"] = "network_analysis"
        elif 'data.csv' in text_lower:
            task_analysis["requires_external_files"] = True
            task_analysis["referenced_files"].append("data.csv")
            task_analysis["task_type"] = "data_analysis"
        
        # Check for visualization requirements
        if any(word in text_lower for word in ['draw', 'plot', 'graph', 'png', 'chart', 'visualization']):
            task_analysis["requires_visualization"] = True
        
        # Check for web scraping requirements
        if any(word in text_lower for word in ['scrape', 'website', 'url', 'web', 'http']):
            task_analysis["requires_web_scraping"] = True
            task_analysis["task_type"] = "web_scraping"
        
        # Check for CSV analysis requirements
        if any(word in text_lower for word in ['csv', 'analyze', 'calculate', 'total', 'sum', 'average', 'count']):
            task_analysis["task_type"] = "csv_analysis"
        
        # Check for embedded data (only if no external files required)
        if not task_analysis["requires_external_files"]:
            # Look for CSV-like patterns
            for i, line in enumerate(lines):
                if ',' in line and not line.startswith('Answer') and not line.startswith('Analyze'):
                    # Check if this looks like CSV data
                    if i + 1 < len(lines) and ',' in lines[i + 1]:
                        task_analysis["has_embedded_data"] = True
                        task_analysis["task_type"] = "embedded_analysis"
                        # Get a preview of the data
                        data_lines = []
                        for j in range(i, min(i + 5, len(lines))):
                            if ',' in lines[j] and not lines[j].startswith('Answer'):
                                data_lines.append(lines[j])
                        task_analysis["data_preview"] = "\n".join(data_lines)
                        break
            
            # Look for table-like patterns
            if not task_analysis["has_embedded_data"]:
                for line in lines:
                    if '|' in line and line.count('|') > 2:
                        task_analysis["has_embedded_data"] = True
                        task_analysis["task_type"] = "embedded_analysis"
                        task_analysis["data_preview"] = line
                        break
        
        return task_analysis
    
    async def _generate_plan_with_llm(self, job_request: JobRequest, file_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to generate execution plan"""
        
        system_prompt = self._get_system_prompt()
        user_prompt = self._create_user_prompt(job_request, file_metadata)
        
        try:
            response = await self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            plan_text = response.choices[0].message.content
            return json.loads(plan_text)
            
        except Exception as e:
            logger.warning(f"Primary model failed, trying fallback: {str(e)}")
            try:
                response = await self.client.chat.completions.create(
                    model=config.OPENAI_FALLBACK_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                plan_text = response.choices[0].message.content
                return json.loads(plan_text)
                
            except Exception as fallback_error:
                raise Exception(f"Both primary and fallback models failed: {str(fallback_error)}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for plan generation"""
        return """You are an expert data analyst AI that creates execution plans for data analysis tasks.

CORE PRINCIPLES:
1. READ and UNDERSTAND the user's specific questions first
2. ANALYZE what data and analysis is actually required
3. CHOOSE actions based on actual needs, not assumptions
4. ORDER actions logically based on data dependencies
5. ADAPT the plan to the specific task requirements

AVAILABLE ACTION TYPES:
- llm_analysis: Analyze embedded data directly using LLM
- load: Load data files into DataFrames
- stats: Basic calculations, counting, statistics
- graph: Network/graph analysis, algorithms
- plot: Create visualizations, charts, graphs
- export: Format and export results
- scrape: Extract data from web pages
- s3_query: Query large datasets in S3/parquet
- sql: Execute SQL queries on loaded data
- api_call: Make API requests
- db_query: Query databases
- sheets_load: Load Google Sheets/Excel
- json_api: Consume JSON APIs
- time_series: Time series analysis
- text_analysis: NLP/text processing
- image_analysis: Extract data from images

PLANNING APPROACH:
- Start with data loading if files are provided
- Add analysis actions based on what questions actually ask for
- Add visualization actions ONLY if charts/graphs are explicitly requested
- Always end with export to format results
- Don't create unnecessary dependencies between actions
- Let the task requirements drive the plan, not predefined templates
- CRITICAL: Be literal about user requests - don't add extra features they didn't ask for

Requirements:
1. Each action must have a unique action_id
2. Actions should be in logical execution order
3. Estimate realistic time for each action (max 300s total)
4. Include proper dependencies between actions
5. Be specific about inputs and outputs
6. Final export action must produce results in the requested format
7. CRITICAL: When generating code, use ONLY pure Python - NO markdown formatting, NO ```python blocks

Return ONLY valid JSON with this structure:
{
  "plan_id": "unique_plan_id",
  "estimated_total_time": 120,
  "actions": [
    {
      "action_id": "action_001",
      "type": "load",
      "description": "Load data file into DataFrame",
      "parameters": {
        "file": "filename.csv"
      },
      "output_variables": ["df"],
      "estimated_time": 30
    }
  ]
}"""

    def _create_user_prompt(self, job_request: JobRequest, file_metadata: Dict[str, Any]) -> str:
        """Create user prompt with job details"""
        # Analyze task requirements
        task_analysis = self._analyze_task_requirements(job_request.questions)
        
        prompt = f"""Task: {job_request.questions}

Available files:
"""
        
        if file_metadata:
            for filename, metadata in file_metadata.items():
                prompt += f"- {filename}: {metadata.get('type', 'unknown')} ({metadata.get('size', 'unknown')} bytes)\n"
                if 'columns' in metadata:
                    prompt += f"  Columns: {', '.join(metadata['columns'][:10])}\n"
                if 'preview' in metadata:
                    prompt += f"  Preview: {metadata['preview'][:200]}...\n"
        else:
            prompt += "No additional files provided.\n"
        
        # Add task-specific guidance
        logger.info(f"🔍 PLANNER DEBUG: Task type detected: {task_analysis['task_type']}")
        logger.info(f"🔍 PLANNER DEBUG: Task analysis: {task_analysis}")
        
        # COMMENTED OUT: Old hardcoded task-specific prompts (kept as backup)
        # These were too rigid and didn't work well for general cases
        # ROLLBACK INSTRUCTIONS: If the new generic prompts fail, uncomment these lines
        # and comment out the "NEW GENERIC PROMPT" section below to restore functionality
        
        # OLD PROMPTS (COMMENTED OUT):
        # if task_analysis["task_type"] == "network_analysis":
        #     prompt += f"""
        # NETWORK ANALYSIS TASK DETECTED:
        # - Task type: {task_analysis["task_type"]}
        # - Required files: {', '.join(task_analysis["referenced_files"])}
        # - Visualization required: {task_analysis["requires_visualization"]}
        # 
        # IMPORTANT: Create a plan with this sequence:
        # 1. load: Load the edges.csv file
        # 2. graph: Perform network analysis (degree, density, shortest paths)
        # 3. plot: Generate network graph and degree histogram
        # 4. export: Format results as JSON with required fields
        # 
        # Use these action types: load, graph, plot, export
        # """
        # # Web scraping removed - not needed for CSV analysis
        # elif task_analysis["has_embedded_data"]:
        #     prompt += f"""
        # EMBEDDED DATA DETECTED:
        # - Task type: {task_analysis["task_type"]}
        # - Data preview: {task_analysis["data_preview"]}
        # 
        # IMPORTANT: Use "llm_analysis" action type since data is embedded in questions.
        # """
        # elif task_analysis["task_type"] == "csv_analysis" or any(f.endswith('.csv') for f in job_request.files):
        #     prompt += f"""
        # CSV ANALYSIS TASK DETECTED:
        # - Task type: {task_analysis["task_type"]}
        # - CSV files: {[f for f in job_request.files if f.endswith('.csv')]}
        # 
        # IMPORTANT: For CSV analysis, use this sequence:
        # 1. load: Load the CSV file(s) into DataFrames
        # 2. stats: Perform calculations (sum, average, count, etc.)
        # 3. export: Format results as requested
        # 
        # Use these action types: load, stats, export
        # DO NOT use SQL for simple CSV calculations - use stats instead.
        # DO NOT use web scraping - this is CSV data analysis, not web data collection.
        # DO NOT mention "Data was scraped successfully" - focus on the actual calculations.
        # 
        # CRITICAL: The export action MUST specify output_files to save the final results.
        # Example export action:
        # {{
        #   "action_id": "action_003",
        #   "type": "export",
        #   "description": "Format the results into a JSON object",
        #   "output_files": ["final_results.json"],
        #   "estimated_time": 30
        # }}
        # """
        # elif task_analysis["task_type"] == "network_analysis" or any(f.endswith('.csv') for f in job_request.files if 'source' in f or 'target' in f):
        #     prompt += f"""
        # NETWORK ANALYSIS TASK DETECTED:
        # - Task type: {task_analysis["task_type"]}
        # - CSV files: {[f for f in job_request.files if f.endswith('.csv')]}
        # 
        # IMPORTANT: For network analysis, use this sequence:
        # 1. load: Load the CSV file(s) into DataFrames
        # 2. graph: Perform network analysis (degree, density, shortest paths)
        # 3. plot: Generate network graph and degree histogram
        # 4. export: Format results as requested
        # 
        # Use these action types: load, graph, plot, export
        # The graph action should analyze network metrics
        # The plot action should generate visualizations
        # DO NOT use web scraping - this is network data analysis, not web data collection.
        # 
        # CRITICAL: The export action MUST specify output_files to save the final results.
        # Example export action:
        # {{
        #   "action_id": "action_004",
        #   "type": "export",
        #   "description": "Format the results into a JSON object",
        #   "output_files": ["final_results.json"],
        #   "estimated_time": 30
        # }}
        # """
        # else:
        #     prompt += f"""
        # GENERAL ANALYSIS TASK:
        # - Task type: {task_analysis["task_type"]}
        # - External files required: {task_analysis["requires_external_files"]}
        # 
        # IMPORTANT: Choose appropriate actions based on the task requirements.
        # """
        
        # NEW GENERIC PROMPT: Intelligent planning based on actual requirements
        prompt += f"""
TASK ANALYSIS:
Questions: {job_request.questions}
Available files: {[f for f in job_request.files] if job_request.files else "None"}

PLANNING INSTRUCTIONS:
1. Read the questions LITERALLY - what is the user actually asking for?
2. Determine what data analysis is required based on the specific questions
3. Choose appropriate actions based on actual needs, not assumptions
4. Order actions logically (data loading → analysis → export)
5. Don't assume specific analysis types - let the questions guide you
6. If questions ask for basic metrics (counts, averages), use 'stats' action
7. If questions ask for complex analysis (algorithms, paths), use 'graph' action
8. If questions ask for charts/visuals, use 'plot' action
9. Always end with 'export' to format results

CRITICAL RULES:
- Only include visualization actions (plot) if the questions EXPLICITLY ask for charts, graphs, or images
- If questions only ask for numerical values, text, or data, do NOT include plot actions
- Be literal about user requests - don't add extra features they didn't ask for
- Don't be "helpful" by adding visualizations unless specifically requested

Create an execution plan that directly addresses the user's questions.
The plan should be intelligent and adaptive, not rigid or template-based.

Generate the JSON plan now:"""
        
        return prompt
    
    def _parse_plan(self, plan_json: Dict[str, Any], task_description: str) -> ExecutionPlan:
        """Parse and validate the JSON plan"""
        try:
            actions = []
            for action_data in plan_json.get("actions", []):
                action = Action(
                    action_id=action_data["action_id"],
                    type=ActionType(action_data["type"]),
                    description=action_data["description"],
                    parameters=action_data.get("parameters", {}),
                    output_files=action_data.get("output_files", []),
                    input_files=action_data.get("input_files", []),
                    output_variables=action_data.get("output_variables", []),
                    input_variables=action_data.get("input_variables", []),
                    dependencies=action_data.get("dependencies", []),
                    estimated_time=action_data.get("estimated_time", 30)
                )
                actions.append(action)
            
            plan = ExecutionPlan(
                plan_id=plan_json.get("plan_id", "default_plan"),
                task_description=task_description,
                actions=actions,
                estimated_total_time=plan_json.get("estimated_total_time", sum(a.estimated_time or 30 for a in actions)),
                metadata=plan_json.get("metadata", {})
            )
            
            return plan
            
        except Exception as e:
            raise Exception(f"Failed to parse execution plan: {str(e)}")
