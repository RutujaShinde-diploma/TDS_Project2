from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json

class ActionType(str, Enum):
    SCRAPE = "scrape"
    LOAD = "load"
    S3_QUERY = "s3_query"
    SQL = "sql"
    STATS = "stats"
    PLOT = "plot"
    GRAPH = "graph"
    EXPORT = "export"
    API_CALL = "api_call"
    DB_QUERY = "db_query"
    SHEETS_LOAD = "sheets_load"
    JSON_API = "json_api"
    TIME_SERIES = "time_series"
    TEXT_ANALYSIS = "text_analysis"
    IMAGE_ANALYSIS = "image_analysis"

class ActionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class Action(BaseModel):
    action_id: str
    type: ActionType
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    input_files: List[str] = Field(default_factory=list)
    output_files: List[str] = Field(default_factory=list)
    input_variables: List[str] = Field(default_factory=list)
    output_variables: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    estimated_time: Optional[int] = None
    
    @validator('action_id')
    def validate_action_id(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("action_id must be a non-empty string")
        return v

class ExecutionPlan(BaseModel):
    plan_id: str
    task_description: str
    actions: List[Action]
    estimated_total_time: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('estimated_total_time')
    def validate_time(cls, v):
        if v > 180:  # 3 minutes max
            raise ValueError("Estimated time cannot exceed 180 seconds")
        return v

class ActionResult(BaseModel):
    action_id: str
    status: ActionStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    generated_code: Optional[str] = None
    retry_count: int = 0
    artifacts: List[str] = Field(default_factory=list)

class ExecutionContext(BaseModel):
    workspace_path: str
    available_files: List[str] = Field(default_factory=list)
    variables: Dict[str, str] = Field(default_factory=dict)  # variable_name: description
    previous_results: List[ActionResult] = Field(default_factory=list)

class JobRequest(BaseModel):
    questions: str
    files: List[str] = Field(default_factory=list)
    output_format: str = "json"
    
class AnalysisAnswer(BaseModel):
    question: str
    answer: str
    value: Optional[str] = None
    unit: Optional[str] = None

class ChartData(BaseModel):
    filename: str
    base64: str
    data_uri: str
    size_bytes: int

class AnalysisResult(BaseModel):
    answers: List[AnalysisAnswer]
    charts_generated: List[str] = Field(default_factory=list)
    charts_data: List[ChartData] = Field(default_factory=list)
    raw_results: List[str] = Field(default_factory=list)

class JobResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Union[List[str], Dict[str, str]]] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None
    plan: Optional[ExecutionPlan] = None
    
class CacheEntry(BaseModel):
    key: str
    value: Any
    timestamp: float
    ttl: int
