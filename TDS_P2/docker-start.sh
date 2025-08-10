#!/bin/bash

# Docker startup script for Data Analyst Agent API

echo "Starting Data Analyst Agent API with Docker..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Build and start the services
echo "Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Check service status
echo "Checking service status..."
docker-compose ps

# Show logs
echo "Showing application logs..."
docker-compose logs -f app
