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

logger = logging.getLogger(__name__)

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
            
            # Try to get from cache first
            cached_plan = await self.cache_manager.get(f"plan:{cache_key}")
            if cached_plan:
                logger.info("Using cached execution plan")
                return ExecutionPlan.parse_obj(cached_plan)
            
            # Analyze uploaded files
            file_metadata = await self._analyze_files(job_request.files, workspace_path)
            
            # Generate plan using LLM
            plan_json = await self._generate_plan_with_llm(job_request, file_metadata)
            
            # Parse and validate the plan
            plan = self._parse_plan(plan_json, job_request.questions)
            
            # Cache the plan
            await self.cache_manager.set(f"plan:{cache_key}", plan.dict(), ttl=config.CACHE_TTL)
            
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
        return """You are an expert data analyst AI that creates detailed execution plans for data analysis tasks.

Given a user's questions and available files (or embedded data), create a JSON execution plan that breaks down the task into discrete actions.

IMPORTANT: Choose action types based on the data source:
- Use "llm_analysis" ONLY when data is embedded directly in the questions
- Use "load", "graph", "plot" when external files are required
- Use "scrape" when web data is needed

Available action types:
- llm_analysis: Analyze embedded data directly using LLM (use this for data embedded in questions)
- load: Load data files into DataFrames (use when external files like edges.csv are needed)
- graph: Analyze network/graph data (use for network analysis)
- plot: Create visualizations (use for generating charts and graphs)
- scrape: Extract data from web pages
- s3_query: Query large datasets in S3/parquet
- sql: Execute SQL queries on loaded data
- stats: Perform statistical computations
- export: Format and export results
- api_call: Make API requests
- db_query: Query databases
- sheets_load: Load Google Sheets/Excel
- json_api: Consume JSON APIs
- time_series: Time series analysis
- text_analysis: NLP/text processing
- image_analysis: Extract data from images

Requirements:
1. Each action must have a unique action_id
2. Actions should be in logical execution order
3. Estimate realistic time for each action (max 180s total)
4. For embedded data, use llm_analysis action
5. For external files, use appropriate file-based actions
6. Include proper dependencies between actions
7. Be specific about inputs and outputs
8. Final export action must produce results in the order questions were asked

Return ONLY valid JSON with this structure:
{
  "plan_id": "unique_plan_id",
  "estimated_total_time": 120,
  "actions": [
    {
      "action_id": "action_001",
      "type": "llm_analysis",
      "description": "Analyze embedded data to answer questions",
      "parameters": {
        "data_type": "csv_embedded",
        "questions": ["question1", "question2"]
      },
      "output_variables": ["analysis_results"],
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
        if task_analysis["task_type"] == "network_analysis":
            prompt += f"""
NETWORK ANALYSIS TASK DETECTED:
- Task type: {task_analysis["task_type"]}
- Required files: {', '.join(task_analysis["referenced_files"])}
- Visualization required: {task_analysis["requires_visualization"]}

IMPORTANT: Create a plan with this sequence:
1. load: Load the edges.csv file
2. graph: Perform network analysis (degree, density, shortest paths)
3. plot: Generate network graph and degree histogram
4. export: Format results as JSON with required fields

Use these action types: load, graph, plot, export
"""
        elif task_analysis["task_type"] == "web_scraping":
            prompt += f"""
WEB SCRAPING TASK DETECTED:
- Task type: {task_analysis["task_type"]}

IMPORTANT: Use scrape action to extract data from websites
"""
        elif task_analysis["has_embedded_data"]:
            prompt += f"""
EMBEDDED DATA DETECTED:
- Task type: {task_analysis["task_type"]}
- Data preview: {task_analysis["data_preview"]}

IMPORTANT: Use "llm_analysis" action type since data is embedded in questions.
"""
        else:
            prompt += f"""
GENERAL ANALYSIS TASK:
- Task type: {task_analysis["task_type"]}
- External files required: {task_analysis["requires_external_files"]}

IMPORTANT: Choose appropriate actions based on the task requirements.
"""
        
        prompt += """
Create an execution plan that:
1. Addresses all questions in the task
2. Uses appropriate action types for each step
3. Follows the recommended action sequence for the task type
4. Handles errors gracefully
5. Completes within 180 seconds
6. Returns results in the requested format

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
