import json
import openai
from typing import List, Dict, Any
import logging
from pathlib import Path
import hashlib

from config import config
from models import JobRequest, ExecutionPlan, Action, ActionType
from utils.cache import CacheManager
from utils.file_analyzer import FileAnalyzer

logger = logging.getLogger(__name__)

class PlannerModule:
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        self.cache_manager = CacheManager()
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

Given a user's questions and available files, create a JSON execution plan that breaks down the task into discrete actions.

Available action types:
- scrape: Extract data from web pages
- load: Load data files into DataFrames
- s3_query: Query large datasets in S3/parquet
- sql: Execute SQL queries on loaded data
- stats: Perform statistical computations
- plot: Create visualizations
- graph: Analyze network/graph data
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
4. Use file references, not raw data
5. Include proper dependencies between actions
6. Be specific about inputs and outputs

Return ONLY valid JSON with this structure:
{
  "plan_id": "unique_plan_id",
  "estimated_total_time": 120,
  "actions": [
    {
      "action_id": "action_001",
      "type": "scrape",
      "description": "Scrape Wikipedia film data",
      "parameters": {
        "url": "https://example.com",
        "target": "table"
      },
      "output_files": ["data.csv"],
      "estimated_time": 30
    }
  ]
}"""

    def _create_user_prompt(self, job_request: JobRequest, file_metadata: Dict[str, Any]) -> str:
        """Create user prompt with job details"""
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
        
        prompt += """
Create an execution plan that:
1. Addresses all questions in the task
2. Uses appropriate action types for each step
3. Handles errors gracefully
4. Completes within 180 seconds
5. Returns results in the requested format

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
