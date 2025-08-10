# Systematic Testing Guide

This guide explains our comprehensive testing system for the Data Analyst Agent API.

## Testing Philosophy

Our testing follows a **bottom-up approach**:
1. **Unit Tests** - Individual components in isolation
2. **Integration Tests** - Component interactions  
3. **API Tests** - Endpoint functionality
4. **End-to-End Tests** - Complete workflows

## Quick Start

### 1. Run Quick Tests (30 seconds)
```bash
python run_tests.py quick
```
Essential tests for rapid feedback during development.

### 2. Run Component Tests (2-3 minutes)
```bash
python run_tests.py component
```
Deep testing of individual components.

### 3. Run Full Test Suite (5-10 minutes)
```bash
python run_tests.py full
```
Comprehensive system testing including E2E workflows.

### 4. Run All Tests (10-15 minutes)
```bash
python run_tests.py all
```
Complete test coverage with detailed reporting.

## Test Categories

### 🔧 Unit Tests
Test individual components in isolation:

- **Configuration Loading**: Environment variables, settings validation
- **Model Validation**: Pydantic model creation and validation
- **Code Validator**: Security checks, AST analysis
- **File Analyzer**: File type detection, metadata extraction
- **Cache Manager**: Set/get operations, TTL handling

**When to run**: After changing any core component

### 🔗 Integration Tests  
Test component interactions:

- **Planner + Validator**: Plan generation and validation pipeline
- **Code Generator + Sandbox**: Code generation and execution pipeline
- **Orchestrator Components**: Full orchestration with retries

**When to run**: After changing component interfaces

### 🌐 API Tests
Test HTTP endpoints:

- **Health Check**: `/health` endpoint functionality
- **Job Submission**: `POST /api/` with file uploads
- **Job Status**: `GET /api/job/{id}` status tracking
- **Error Handling**: Invalid requests, missing files

**When to run**: After changing API layer

### 🎯 End-to-End Tests
Test complete workflows:

- **Simple Analysis**: Basic data generation and analysis
- **CSV Analysis**: File upload, processing, and results
- **Web Scraping**: External data fetching and analysis
- **Error Recovery**: Handling and retry scenarios

**When to run**: Before deployment, after major changes

## Test Structure

```
tests/
├── __init__.py           # Test package
├── test_runner.py        # Complete systematic test suite
├── quick_test.py         # Essential tests for rapid feedback
├── component_tests.py    # Deep component testing
└── run_tests.py         # Test orchestrator (main entry point)
```

## Usage Examples

### Development Workflow
```bash
# During development - quick feedback
python run_tests.py quick

# Before committing - component verification
python run_tests.py component

# Before merging - full verification
python run_tests.py full
```

### CI/CD Integration
```bash
# In CI pipeline
python run_tests.py all --continue-on-failure
```

### Debugging Failed Tests
```bash
# Run specific test type with detailed output
python run_tests.py component

# Check API server status
python run_tests.py quick --no-api-check
```

## Test Requirements

### Prerequisites
1. **API Server Running**: For API and E2E tests
   ```bash
   python start_local.py
   ```

2. **Environment Setup**: OpenAI API key configured
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

3. **Dependencies**: All packages installed
   ```bash
   pip install -r requirements_simple.txt
   ```

### Optional Requirements
- **Redis**: For cache testing (uses in-memory fallback if unavailable)
- **Docker**: For container-based execution (uses subprocess fallback)

## Understanding Test Results

### ✅ Success Indicators
- All tests passing
- Response times within limits
- No error messages in logs
- API server responding correctly

### ❌ Failure Patterns

**Import Errors**:
```
❌ Import test failed: No module named 'xyz'
```
*Solution*: Install missing dependencies

**API Connection Errors**:
```
❌ Cannot connect to API: Connection refused
```
*Solution*: Start the API server with `python start_local.py`

**Configuration Errors**:
```
❌ OPENAI_API_KEY environment variable not set
```
*Solution*: Set your OpenAI API key

**Timeout Errors**:
```
❌ Analysis timed out
```
*Solution*: Check OpenAI API connectivity and model availability

## Test Configuration

### Environment Variables
```bash
# Test-specific settings
OPENAI_API_KEY=your-key-here          # Required for LLM tests
TEST_TIMEOUT=300                      # Test timeout in seconds
TEST_SKIP_E2E=false                   # Skip end-to-end tests
TEST_API_URL=http://localhost:8000    # API base URL
```

### Command Line Options
```bash
python run_tests.py full --no-api-check           # Skip API server check
python run_tests.py all --continue-on-failure     # Continue despite failures
```

## Performance Benchmarks

### Expected Test Times
- **Quick Tests**: 15-30 seconds
- **Component Tests**: 1-3 minutes  
- **API Tests**: 30-60 seconds
- **E2E Tests**: 3-8 minutes
- **Full Suite**: 5-12 minutes

### Performance Indicators
- API response time < 2 seconds
- Code execution < 30 seconds
- File processing < 5 seconds
- Cache operations < 100ms

## Debugging Guide

### Common Issues

1. **Import Errors**
   - Check Python path configuration
   - Verify all dependencies installed
   - Ensure project structure is correct

2. **API Server Issues**  
   - Verify server is running on correct port
   - Check for port conflicts
   - Review server logs for errors

3. **OpenAI API Issues**
   - Verify API key is valid
   - Check rate limits and quotas
   - Test with simpler requests first

4. **File System Issues**
   - Check workspace directory permissions
   - Verify temporary file creation
   - Ensure sufficient disk space

### Debug Commands
```bash
# Check system status
python -c "from tests.quick_test import *; asyncio.run(quick_unit_tests())"

# Test individual components
python -c "from sandbox import CodeValidator; v = CodeValidator(); print(v.validate_code('print(1)'))"

# Check API manually
curl http://localhost:8000/health
```

## Contributing Tests

### Adding New Tests

1. **Unit Tests**: Add to `component_tests.py`
2. **Integration Tests**: Add to `test_runner.py` integration section
3. **API Tests**: Add to `test_runner.py` API section
4. **E2E Tests**: Add to `test_runner.py` E2E section

### Test Patterns
```python
async def test_new_feature(self) -> TestResult:
    """Test new feature functionality"""
    start_time = time.time()
    
    try:
        # Test implementation
        assert condition, "Error message"
        
        duration = time.time() - start_time
        return TestResult("New Feature", True, "Success message", duration)
        
    except Exception as e:
        duration = time.time() - start_time
        return TestResult("New Feature", False, str(e), duration)
```

### Best Practices
- Use descriptive test names
- Include both positive and negative test cases
- Clean up resources (files, connections)
- Provide meaningful error messages
- Test edge cases and error conditions

## Continuous Integration

### GitHub Actions Example
```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: pip install -r requirements_simple.txt
      - name: Run tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python run_tests.py all --continue-on-failure
```

## Monitoring and Metrics

### Test Metrics to Track
- Test pass rate over time
- Average test execution time
- Flaky test identification
- Coverage of new features

### Health Indicators
- 95%+ test pass rate
- < 10% increase in test time
- Zero critical component failures
- API uptime > 99%

This systematic testing approach ensures robust, reliable operation of the Data Analyst Agent API.
