from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import aiofiles
import os
import uuid
import time
import logging
import hashlib
from typing import List, Optional
import asyncio
from pathlib import Path

from config import config
from models import JobRequest, JobResponse, ExecutionPlan
from planner import PlannerModule
from validator import PlanValidator
from orchestrator import Orchestrator
from utils.logger import setup_logger
from utils.cache import CacheManager
from utils.job_storage import job_storage

# Setup logging
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst Agent API",
    description="API that uses LLMs to source, prepare, analyze, and visualize data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware - ENABLED for debugging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging"""
    logger.info(f"=== INCOMING REQUEST DEBUG ===")
    logger.info(f"Method: {request.method}")
    logger.info(f"URL: {request.url}")
    logger.info(f"Headers: {dict(request.headers)}")
    
    # Log form data for POST requests
    if request.method == "POST":
        try:
            form_data = await request.form()
            logger.info(f"Form data keys: {list(form_data.keys())}")
            for key, value in form_data.items():
                if hasattr(value, 'filename'):
                    logger.info(f"File field '{key}': filename={value.filename}, size={value.size}")
                else:
                    logger.info(f"Form field '{key}': {value}")
        except Exception as e:
            logger.warning(f"Could not parse form data: {e}")
    
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    logger.info(f"=== END REQUEST DEBUG ===")
    return response

# Initialize components
planner = PlannerModule()
validator = PlanValidator()
orchestrator = Orchestrator()
cache_manager = CacheManager()

# Note: Job storage is now handled by persistent job_storage manager

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Data Analyst Agent API")
    
    # Create workspace directory
    os.makedirs(config.WORKSPACE_DIR, exist_ok=True)
    
    # Initialize cache
    await cache_manager.initialize()
    
    logger.info("API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Data Analyst Agent API")
    await cache_manager.close()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Data Analyst Agent API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {
            "planner": "ok",
            "validator": "ok", 
            "orchestrator": "ok",
            "cache": "ok" if cache_manager.is_connected() else "disconnected",
            "job_storage": "ok"
        }
    }

@app.post("/test-upload")
async def test_file_upload(questions: UploadFile = File(...)):
    """Test endpoint to verify file uploads are working"""
    try:
        if not questions:
            return {"error": "No file received"}
        
        content = await questions.read()
        text = content.decode('utf-8')
        
        return {
            "success": True,
            "filename": questions.filename,
            "size": questions.size,
            "content_length": len(text),
            "content_preview": text[:100] + "..." if len(text) > 100 else text
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/storage/stats")
async def get_storage_stats():
    """Get job storage statistics for debugging"""
    return job_storage.get_storage_stats()

@app.post("/api/cache/clear")
async def clear_all_caches():
    """Clear all caches (useful for testing and debugging)"""
    try:
        # Clear planner cache
        await cache_manager.delete("planner_cache")
        
        # Clear LLM cache
        await cache_manager.delete("llm_cache")
        
        # Clear code cache
        await cache_manager.delete("code_cache")
        
        # Clear all cached plans (this is a bit aggressive but effective)
        await cache_manager.delete("plan:*")
        
        logger.info("All caches cleared successfully")
        return {"message": "All caches cleared successfully", "timestamp": time.time()}
        
    except Exception as e:
        logger.error(f"Error clearing caches: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear caches: {str(e)}")

@app.get("/api/cache/status")
async def get_cache_status():
    """Get cache status and statistics"""
    try:
        # Get cache connection status
        redis_status = "connected" if cache_manager.is_connected() else "disconnected"
        
        # Get some cache statistics
        cache_stats = {
            "redis_status": redis_status,
            "cache_manager_connected": cache_manager.is_connected(),
            "timestamp": time.time()
        }
        
        return cache_stats
        
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache status: {str(e)}")

@app.get("/api/cache/debug")
async def get_cache_debug_info():
    """Get detailed cache debugging information"""
    try:
        # Get detailed cache stats
        cache_stats = await cache_manager.get_cache_stats()
        
        # Search for plan-related cache keys
        plan_keys = await cache_manager.search_cache_keys("plan:*")
        
        # Search for LLM-related cache keys
        llm_keys = await cache_manager.search_cache_keys("llm:*")
        
        # Search for code-related cache keys
        code_keys = await cache_manager.search_cache_keys("code:*")
        
        debug_info = {
            "cache_stats": cache_stats,
            "plan_cache_keys": plan_keys,
            "llm_cache_keys": llm_keys,
            "code_cache_keys": code_keys,
            "total_cached_items": len(plan_keys) + len(llm_keys) + len(code_keys),
            "timestamp": time.time()
        }
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Error getting cache debug info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache debug info: {str(e)}")

@app.post("/api/cache/clear/plan")
async def clear_plan_cache(questions: str = Form(...)):
    """Clear cache for a specific plan based on questions content"""
    try:
        # Generate cache key for the specific questions
        cache_key = hashlib.md5(questions.encode()).hexdigest()
        
        # Clear any cached plans for this content
        await cache_manager.delete(f"plan:{cache_key}")
        
        logger.info(f"Cleared plan cache for questions: {questions[:50]}...")
        return {"message": "Plan cache cleared", "cache_key": cache_key, "timestamp": time.time()}
        
    except Exception as e:
        logger.error(f"Error clearing plan cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear plan cache: {str(e)}")

@app.post("/api/")
async def analyze_data(
    request: Request,
    questions: UploadFile = File(None, description="questions.txt file with task description"),
    questions_txt: UploadFile = File(None, description="Alternative field name for questions.txt"),
    files: List[UploadFile] = File(default=[], description="Optional data files"),
    bypass_cache: bool = Form(default=False, description="Bypass cache and force fresh execution")
):
    """
    Main endpoint for data analysis requests
    
    Accepts:
    - questions.txt (required): Contains the analysis questions/tasks
    - Additional files (optional): Multiple data files, images, etc.
    - bypass_cache (optional): Force fresh execution ignoring cache
    
    Returns answers directly in format: {"answer1": "value1", "answer2": "value2", ...}
    
    Example usage:
    curl "https://app.example.com/api/" -F "questions=@questions.txt" -F "files=@image.png" -F "files=@data.csv"
    """
    start_time = time.time()
    job_id = str(uuid.uuid4())
    
    logger.info(f"Starting job {job_id} (bypass_cache: {bypass_cache})")
    
    try:
        # Handle both field names for questions file
        questions_file = questions or questions_txt
        
        # Enhanced debug logging for file upload
        logger.info(f"=== FILE UPLOAD DEBUG ===")
        logger.info(f"Questions field received: {'questions' if questions else 'questions_txt' if questions_txt else 'None'}")
        logger.info(f"Questions file object: {questions_file}")
        logger.info(f"Questions filename: {questions_file.filename if questions_file else 'None'}")
        logger.info(f"Questions file size: {questions_file.size if questions_file else 'None'}")
        logger.info(f"Questions content type: {questions_file.content_type if questions_file else 'None'}")
        
        # Log all request details
        logger.info(f"Request method: {request.method if 'request' in locals() else 'Unknown'}")
        logger.info(f"Request URL: {request.url if 'request' in locals() else 'Unknown'}")
        
        # Validate that questions file was actually received
        if not questions_file:
            raise HTTPException(status_code=400, detail="No questions file received. Please send a file with field name 'questions' or 'questions_txt'")
        
        # Validate questions file
        if not questions_file.filename or not questions_file.filename.endswith('.txt'):
            logger.error(f"Invalid questions file: {questions_file.filename}")
            raise HTTPException(status_code=400, detail="questions file must be a .txt file")
        
        # Create job workspace
        job_workspace = Path(config.WORKSPACE_DIR) / job_id
        job_workspace.mkdir(exist_ok=True)
        
        # Save questions file
        try:
            logger.info(f"📁 Reading questions file for job {job_id}")
            # Add timeout for file reading
            questions_content = await asyncio.wait_for(
                questions_file.read(),
                timeout=10  # 10 seconds timeout for file reading
            )
            if not questions_content:
                raise HTTPException(status_code=400, detail="Questions file is empty")
            
            questions_text = questions_content.decode('utf-8')
            logger.info(f"✅ Successfully read questions file for job {job_id}, content length: {len(questions_text)}")
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ Questions file reading timed out for job {job_id}")
            raise HTTPException(status_code=408, detail="Questions file reading timed out")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Questions file must be valid UTF-8 text")
        except Exception as e:
            logger.error(f"Error reading questions file for job {job_id}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to read questions file: {str(e)}")
        
        questions_path = job_workspace / "questions.txt"
        try:
            logger.info(f"📁 Writing questions file for job {job_id}")
            async with aiofiles.open(questions_path, 'w') as f:
                await asyncio.wait_for(
                    f.write(questions_text),
                    timeout=10  # 10 seconds timeout for file writing
                )
            logger.info(f"✅ Questions file written successfully for job {job_id}")
        except asyncio.TimeoutError:
            logger.error(f"⏰ Questions file writing timed out for job {job_id}")
            raise HTTPException(status_code=408, detail="Questions file writing timed out")
        except Exception as e:
            logger.error(f"Error writing questions file for job {job_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to write questions file: {str(e)}")
        
        # Save additional files
        uploaded_files = []
        if files:
            logger.info(f"📁 Processing {len(files)} additional files for job {job_id}")
            for i, file in enumerate(files):
                try:
                    logger.info(f"📁 Processing file {i+1}/{len(files)}: {file.filename}")
                    if file.size > config.MAX_FILE_SIZE:
                        raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds size limit")
                    
                    file_path = job_workspace / file.filename
                    content = await asyncio.wait_for(
                        file.read(),
                        timeout=30  # 30 seconds timeout per file
                    )
                    async with aiofiles.open(file_path, 'wb') as f:
                        await asyncio.wait_for(
                            f.write(content),
                            timeout=30  # 30 seconds timeout per file
                        )
                    
                    uploaded_files.append(file.filename)
                    logger.info(f"✅ File {file.filename} processed successfully for job {job_id}")
                except asyncio.TimeoutError:
                    logger.error(f"⏰ File {file.filename} processing timed out for job {job_id}")
                    raise HTTPException(status_code=408, detail=f"File {file.filename} processing timed out")
                except Exception as e:
                    logger.error(f"Error processing file {file.filename} for job {job_id}: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Failed to process file {file.filename}: {str(e)}")
        
        # Create job request
        job_request = JobRequest(
            questions=questions_text,
            files=uploaded_files
        )
        
        # Process job synchronously (wait for completion)
        logger.info(f"Processing job {job_id} synchronously")
        
        # Set execution timeout to prevent hanging (5 minutes max)
        execution_timeout = 300  # 5 minutes in seconds
        logger.info(f"⏱️ Setting execution timeout to {execution_timeout} seconds for job {job_id}")
        
        # Generate execution plan (with cache bypass option)
        if bypass_cache:
            logger.info("Bypassing cache for plan generation")
            # Clear any existing cached plan for this request
            cache_key = hashlib.md5(questions_text.encode()).hexdigest()
            await cache_manager.delete(f"plan:{cache_key}")
        
        # 1. PLAN GENERATION - Wrap with explicit error handling and timeout
        try:
            logger.info(f"🔄 Starting plan generation for job {job_id}")
            # Add timeout for plan generation
            plan = await asyncio.wait_for(
                planner.create_plan(job_request, str(job_workspace)),
                timeout=60  # 1 minute timeout for plan generation
            )
            if not plan:
                logger.error(f"❌ Plan generation returned None for job {job_id}")
                raise HTTPException(status_code=500, detail="Failed to create execution plan")
            logger.info(f"✅ Plan created successfully for job {job_id}: {len(plan.actions)} actions")
            logger.info(f"📋 Plan actions: {[action.action_type for action in plan.actions]}")
        except asyncio.TimeoutError:
            logger.error(f"⏰ Plan generation timed out after 60 seconds for job {job_id}")
            raise HTTPException(status_code=408, detail="Plan generation timed out")
        except Exception as e:
            logger.exception(f"❌ CRASH during plan creation for job {job_id}")
            raise HTTPException(status_code=500, detail=f"Error during plan creation: {str(e)}")
        
        # 2. PLAN VALIDATION - Wrap with explicit error handling and timeout
        try:
            logger.info(f"🔍 Starting plan validation for job {job_id}")
            # Add timeout for plan validation
            validation_result = await asyncio.wait_for(
                validator.validate_plan(plan),
                timeout=30  # 30 seconds timeout for validation
            )
            if not validation_result.is_valid:
                logger.error(f"❌ Plan validation failed for job {job_id}: {validation_result.errors}")
                raise HTTPException(status_code=400, detail=f"Invalid plan: {validation_result.errors}")
            logger.info(f"✅ Plan validation successful for job {job_id}")
        except asyncio.TimeoutError:
            logger.error(f"⏰ Plan validation timed out after 30 seconds for job {job_id}")
            raise HTTPException(status_code=408, detail="Plan validation timed out")
        except Exception as e:
            logger.exception(f"❌ CRASH during plan validation for job {job_id}")
            raise HTTPException(status_code=500, detail=f"Error during plan validation: {str(e)}")
        
        # 3. PLAN EXECUTION - Wrap with explicit error handling and timeout
        try:
            logger.info(f"🚀 Starting plan execution for job {job_id}")
            # Add timeout for plan execution (remaining time after plan generation and validation)
            remaining_timeout = execution_timeout - 90  # 90 seconds already used
            result = await asyncio.wait_for(
                orchestrator.execute_plan(plan, str(job_workspace)),
                timeout=remaining_timeout
            )
            logger.info(f"✅ Plan execution completed for job {job_id}")
            logger.info(f"📊 Execution result type: {type(result)}")
            if isinstance(result, dict):
                logger.info(f"�� Result keys: {list(result.keys())}")
            elif isinstance(result, list):
                logger.info(f"📊 Result list length: {len(result)}")
            else:
                logger.info(f"📊 Result: {str(result)[:200]}...")
        except asyncio.TimeoutError:
            logger.error(f"⏰ Plan execution timed out after {remaining_timeout} seconds for job {job_id}")
            raise HTTPException(status_code=408, detail="Plan execution timed out")
        except Exception as e:
            logger.exception(f"❌ CRASH during plan execution for job {job_id}")
            raise HTTPException(status_code=500, detail=f"Error during plan execution: {str(e)}")
        
        execution_time = time.time() - start_time
        
        # Log final job summary
        logger.info(f"🎯 Job Summary: ID={job_id}, Time={execution_time:.2f}s, ResultType={type(result)}")
        
        # Return answers directly
        if isinstance(result, dict):
            # Already in answer1, answer2 format
            logger.info(f"✅ Returning dict result for job {job_id}")
            return result
        elif isinstance(result, list):
            # Convert list to answer1, answer2 format
            answers = {}
            for i, answer in enumerate(result, 1):
                answers[f"answer{i}"] = str(answer)
            logger.info(f"✅ Returning list result converted to answers for job {job_id}")
            return answers
        else:
            # Single result
            logger.info(f"✅ Returning single result for job {job_id}")
            return {"answer1": str(result)}
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/job/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get status and results of a specific job"""
    job_record = job_storage.get_job(job_id)
    
    if not job_record:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Convert job record to response format
    # Handle plan conversion from dict to ExecutionPlan if needed
    plan_obj = None
    if job_record.plan:
        try:
            if isinstance(job_record.plan, dict):
                # Convert dict back to ExecutionPlan object
                plan_obj = ExecutionPlan(**job_record.plan)
            else:
                plan_obj = job_record.plan
        except Exception as e:
            logger.warning(f"Failed to convert plan for job {job_id}: {e}")
            # Continue without plan to avoid 500 error
    
    return JobResponse(
        job_id=job_record.job_id,
        status=job_record.status,
        execution_time=job_record.execution_time,
        result=job_record.result,
        error=job_record.error,
        plan=plan_obj
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
