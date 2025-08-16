# Local Development Guide

This guide shows how to run the Data Analyst Agent API locally without Docker.

## Quick Start

### 1. Setup Environment

```bash
# Navigate to project directory
cd TDS_P2

# Run setup script (installs requirements, creates .env)
python setup_local.py
```

The setup script will:
- Install Python requirements
- Create `.env` file with your OpenAI API key
- Create necessary directories
- Configure for local execution (no Docker/Redis needed)

### 2. Start the API Server

```bash
python start_local.py
```

This will:
- Check requirements and configuration
- Start the FastAPI server on http://localhost:8000
- Use in-memory cache (no Redis required)
- Use subprocess execution (no Docker required)

### 3. Test the API

In another terminal:

```bash
# Simple test
python test_local.py

# Full test suite
python test_example.py
```

## Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Create Environment File

Create `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_FALLBACK_MODEL=gpt-3.5-turbo
REDIS_URL=memory://localhost
MAX_EXECUTION_TIME=180
LOG_LEVEL=INFO
USE_DOCKER=false
```

### 3. Create Directories

```bash
mkdir -p workspace logs
```

### 4. Start Server

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Local Configuration

### Key Differences from Docker Setup

1. **No Docker**: Code executes in subprocess with security restrictions
2. **No Redis**: Uses in-memory caching
3. **Local Files**: All workspace files stored in `./workspace/`
4. **Development Mode**: Auto-reload enabled for code changes

### Environment Variables for Local

- `USE_DOCKER=false` - Disables Docker execution
- `REDIS_URL=memory://localhost` - Uses in-memory cache
- Other variables same as Docker setup

## Testing Examples

### Test 1: Simple Analysis
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "questions=@questions.txt"
```

### Test 2: CSV Analysis
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "questions=@questions.txt" \
  -F "files=@sample.csv"
```

### Test 3: Multiple Files
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "questions=@questions.txt" \
  -F "files=@sample.csv" \
  -F "files=@image.png" \
  -F "files=@data.json"
```

### Test 4: Check Status
```bash
curl "http://localhost:8000/api/job/{job_id}"
```

## API Endpoints

- **Health Check**: `GET /health`
- **Submit Job**: `POST /api/`
- **Job Status**: `GET /api/job/{job_id}`
- **API Docs**: `GET /docs`

## Directory Structure

```
TDS_P2/
├── workspace/          # Job workspaces (created automatically)
├── logs/              # Application logs
├── main.py            # FastAPI application
├── start_local.py     # Local development server
├── test_local.py      # Simple local tests
├── setup_local.py     # Environment setup
└── ...
```

## Troubleshooting

### Common Issues

1. **Missing OpenAI Key**
   ```
   Error: OPENAI_API_KEY not set
   Solution: Add to .env file or export as environment variable
   ```

2. **Port Already in Use**
   ```
   Error: [Errno 98] Address already in use
   Solution: Kill process on port 8000 or use different port
   ```

3. **Module Import Errors**
   ```
   Error: No module named 'fastapi'
   Solution: pip install -r requirements.txt
   ```

4. **Permission Errors**
   ```
   Error: Permission denied
   Solution: Check file permissions and workspace directory access
   ```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python start_local.py
```

### View Logs

```bash
# Real-time logs
tail -f logs/app.log

# Or check console output when running start_local.py
```

## Performance Notes

### Local vs Docker

- **Startup**: Faster (no container overhead)
- **Execution**: Similar performance
- **Security**: Reduced isolation (subprocess vs container)
- **Dependencies**: Must install locally

### Optimization Tips

1. **Cache Warming**: First request slower due to cache misses
2. **Model Choice**: Use `gpt-3.5-turbo` for faster responses during testing
3. **File Size**: Keep test files small for faster processing

## Development Workflow

1. **Make Changes**: Edit code files
2. **Auto-Reload**: Server automatically restarts
3. **Test**: Run `python test_local.py`
4. **Debug**: Check logs and console output
5. **Iterate**: Repeat

## Security Considerations

### Local Security

- Code execution happens in subprocess with restricted environment
- No network access except to allowed domains
- File system access limited to workspace directory
- Static code analysis still active

### Production Deployment

For production, use Docker setup:
```bash
docker-compose up -d
```

This provides better isolation and security.
