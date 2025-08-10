#!/bin/bash

# Start script for Data Analyst Agent API

set -e

echo "Starting Data Analyst Agent API..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Please create one with your OpenAI API key."
    echo "You can copy .env.example and modify it."
    echo ""
    echo "Required environment variables:"
    echo "  OPENAI_API_KEY=your_openai_api_key_here"
    echo ""
fi

# Create necessary directories
mkdir -p workspace logs

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Start services with Docker Compose
echo "Starting services with Docker Compose..."
docker-compose up -d

echo ""
echo "Services starting..."
echo ""
echo "API will be available at: http://localhost:8000"
echo "Health check: http://localhost:8000/health"
echo "API docs: http://localhost:8000/docs"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f app"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""

# Wait a moment and check if services are running
sleep 3

if docker-compose ps | grep -q "Up"; then
    echo "✅ Services are running successfully!"
    
    # Optional: run health check
    echo ""
    echo "Checking API health..."
    
    for i in {1..10}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ API is healthy and ready!"
            break
        else
            echo "⏳ Waiting for API to be ready... ($i/10)"
            sleep 2
        fi
    done
    
else
    echo "❌ Some services failed to start. Check logs:"
    echo "  docker-compose logs"
fi
