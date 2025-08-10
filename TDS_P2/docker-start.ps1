# Docker startup script for Data Analyst Agent API (PowerShell)

Write-Host "Starting Data Analyst Agent API with Docker..." -ForegroundColor Green

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "Error: Docker is not running. Please start Docker first." -ForegroundColor Red
    exit 1
}

# Build and start the services
Write-Host "Building and starting services..." -ForegroundColor Yellow
docker-compose up --build -d

# Wait for services to be ready
Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service status
Write-Host "Checking service status..." -ForegroundColor Yellow
docker-compose ps

# Show logs
Write-Host "Showing application logs..." -ForegroundColor Yellow
docker-compose logs -f app
