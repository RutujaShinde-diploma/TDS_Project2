# Data Analyst Agent API

A powerful API that uses Large Language Models (LLMs) to autonomously source, prepare, analyze, and visualize data. The system accepts arbitrary data analysis requests via natural language and returns comprehensive answers within 3 minutes.

## Features

- **LLM-Driven Planning**: Creates detailed execution plans for complex analysis tasks
- **Multi-Action Support**: Scraping, loading, statistical analysis, visualization, and more
- **Secure Execution**: Sandboxed code execution with Docker isolation
- **Error Recovery**: Automatic retry and code repair mechanisms
- **Caching**: Intelligent caching for performance optimization
- **Comprehensive Logging**: Full audit trail of all operations

## Architecture

### Core Components

1. **API Endpoint (FastAPI)**: Handles POST requests with file uploads
2. **Planner Module**: LLM-driven task analysis and plan generation
3. **Plan Validator**: Safety checks and schema validation
4. **Orchestrator**: Sequential action execution with retry logic
5. **Sandbox Environment**: Secure Docker-based code execution
6. **Code Generator**: LLM-powered Python code generation
7. **Cache Manager**: Redis-based caching for performance

### Supported Action Types

- `scrape`: Extract data from web pages
- `load`: Load data files into DataFrames
- `sql`: Execute SQL queries on loaded data
- `stats`: Perform statistical computations
- `plot`: Create visualizations and charts
- `export`: Format and export final results
- `api_call`: Make external API requests
- `time_series`: Time series analysis
- `text_analysis`: NLP and text processing

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Redis (included in docker-compose)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TDS_P2
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

3. Start the services:
```bash
docker-compose up -d
```

4. The API will be available at `http://localhost:8000`

### Development Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Start Redis:
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

3. Set environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
export REDIS_URL="redis://localhost:6379"
```

4. Run the application:
```bash
uvicorn main:app --reload
```

## Usage

### API Endpoint

The main endpoint accepts POST requests with file uploads:

```bash
curl -X POST "http://localhost:8000/api/" \
  -F "questions=@questions.txt" \
  -F "data=@data.csv"
```

### Request Format

- `questions.txt` (required): Contains the analysis questions/tasks
- Additional files (optional): Data files, images, etc.

### Response Format

The API returns a JSON response with:
- `job_id`: Unique identifier for the request
- `status`: Current processing status
- `result`: Final analysis results (when completed)
- `execution_time`: Total processing time
- `error`: Error message (if failed)

### Example Request

**questions.txt:**
```
Analyze the sales data and answer these questions:
1. What are the top 5 products by revenue?
2. What is the monthly sales trend?
3. Create a visualization showing revenue by category.
Return results as a JSON array of strings.
```

**Response:**
```json
{
  "job_id": "12345-abcde",
  "status": "completed",
  "result": [
    "Top 5 products: Product A ($50K), Product B ($45K), ...",
    "Monthly trend shows 15% growth from Jan to Dec",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
  ],
  "execution_time": 87.5
}
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key (required)
- `OPENAI_MODEL`: Primary model (default: gpt-4)
- `OPENAI_FALLBACK_MODEL`: Fallback model (default: gpt-3.5-turbo)
- `MAX_EXECUTION_TIME`: Maximum execution time in seconds (default: 180)
- `REDIS_URL`: Redis connection URL
- `LOG_LEVEL`: Logging level (default: INFO)

### Safety Configuration

The system includes several safety measures:

- **Blocked imports**: `os`, `subprocess`, `sys`, etc.
- **Code validation**: AST analysis for dangerous patterns
- **Domain whitelist**: Only approved domains for scraping
- **Resource limits**: CPU, memory, and time constraints
- **Sandbox isolation**: Docker containers with restricted access

## API Documentation

### Health Check

```
GET /health
```

Returns system health status and component availability.

### Job Status

```
GET /api/job/{job_id}
```

Returns the current status and results of a specific job.

## Development

### Adding New Action Types

1. Add the action type to `models.py`:
```python
class ActionType(str, Enum):
    NEW_ACTION = "new_action"
```

2. Add validation logic in `validator.py`
3. Add code generation instructions in `code_generator.py`
4. Update the planner prompts to include the new action type

### Testing

Run the test suite:
```bash
pytest tests/
```

### Monitoring

The application includes comprehensive logging:
- Structured JSON logs
- Request/response tracking
- Performance metrics
- Error tracking

Logs are available in the `logs/` directory.

## Security

### Sandbox Security

- Docker container isolation
- No network access except allowed domains
- Restricted file system access
- Resource limits (CPU, memory, time)
- Static code analysis before execution

### Input Validation

- File type restrictions
- Size limits
- Content validation
- SQL injection prevention
- XSS protection

## Performance

### Optimization Features

- Redis caching for LLM responses
- Code pattern caching
- File metadata caching
- Parallel processing where possible
- Efficient memory management

### Scaling

The system can be scaled horizontally:
- Multiple API instances behind a load balancer
- Shared Redis cache
- Distributed file storage
- Background job processing

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**: Check API key and rate limits
2. **Docker Issues**: Ensure Docker daemon is running
3. **Memory Errors**: Increase Docker memory limits
4. **Timeout Errors**: Adjust `MAX_EXECUTION_TIME`

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
```

### Health Checks

Monitor the health endpoint:
```bash
curl http://localhost:8000/health
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs
3. Open an issue on GitHub
