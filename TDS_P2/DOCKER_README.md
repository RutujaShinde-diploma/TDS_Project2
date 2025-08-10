# Docker Deployment Guide

This guide explains how to deploy the Data Analyst Agent API using Docker.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose installed
- At least 4GB of available RAM
- OpenAI API key

## Quick Start

### 1. Start the Application

**Windows PowerShell:**
```powershell
.\docker-start.ps1
```

**Linux/macOS:**
```bash
./docker-start.sh
```

**Manual:**
```bash
docker-compose up --build -d
```

### 2. Check Status

```bash
docker-compose ps
```

### 3. View Logs

```bash
docker-compose logs -f app
```

## Configuration

### Environment Variables

The application uses `docker.env` file for configuration. Key variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Primary model (default: gpt-4)
- `OPENAI_FALLBACK_MODEL`: Fallback model (default: gpt-3.5-turbo)
- `MAX_EXECUTION_TIME`: Maximum execution time in seconds (default: 180)
- `LOG_LEVEL`: Logging level (default: INFO)

### Ports

- **API**: 8000 (http://localhost:8000)
- **Redis**: 6379 (internal only)

## Services

### App Service
- **Image**: Built from local Dockerfile
- **Port**: 8000
- **Health Check**: `/health` endpoint
- **Volumes**: 
  - `./workspace` → `/app/workspace`
  - `./logs` → `/app/logs`

### Redis Service
- **Image**: redis:7-alpine
- **Port**: 6379 (internal)
- **Persistence**: Redis data volume
- **Health Check**: Redis ping command

## Health Checks

The application includes health checks for both services:

- **App**: Checks `/health` endpoint every 30s
- **Redis**: Pings Redis every 30s

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Check what's using port 8000
   netstat -ano | findstr :8000
   
   # Stop the service using the port
   docker-compose down
   ```

2. **API key not working**
   - Verify `docker.env` contains correct API key
   - Check logs: `docker-compose logs app`

3. **Redis connection issues**
   ```bash
   # Check Redis logs
   docker-compose logs redis
   
   # Restart Redis
   docker-compose restart redis
   ```

### Debug Commands

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs app
docker-compose logs redis

# Access app container
docker-compose exec app bash

# Check service status
docker-compose ps

# Restart services
docker-compose restart

# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v
```

## Development

### Rebuilding

```bash
# Rebuild and restart
docker-compose up --build -d

# Force rebuild (no cache)
docker-compose build --no-cache
```

### Testing

```bash
# Run tests in container
docker-compose exec app python -m pytest

# Run specific test
docker-compose exec app python test_simple.py
```

## Production Considerations

1. **Security**: Don't expose Redis port externally
2. **Scaling**: Use Docker Swarm or Kubernetes for production
3. **Monitoring**: Add Prometheus/Grafana for metrics
4. **Backup**: Regular Redis data backups
5. **Logs**: Configure log rotation and external logging

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /api/` - Main analysis endpoint
- `GET /api/job/{job_id}` - Job status
- `GET /api/cache/status` - Cache status
- `POST /api/cache/clear` - Clear cache

## Support

For issues or questions:
1. Check the logs: `docker-compose logs app`
2. Verify configuration in `docker.env`
3. Check Docker Desktop is running
4. Ensure sufficient system resources
