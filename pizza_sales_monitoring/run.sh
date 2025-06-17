#!/bin/bash

# Check Docker permissions first
echo "🔍 Checking Docker permissions..."
if ! docker ps &>/dev/null; then
    echo "❌ Docker permission denied. Fixing..."

    # Add user to docker group
    sudo usermod -aG docker $USER

    # Fix socket permissions
    sudo chmod 666 /var/run/docker.sock

    # Restart Docker
    sudo systemctl restart docker

    # Reload groups
    newgrp docker

    echo "✅ Docker permissions fixed. Please run script again."
    exit 0
fi

echo "✅ Docker permissions OK"

# Check current directory and files
echo "📁 Current directory: $(pwd)"
echo "📄 Checking required files..."

if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found!"
    exit 1
fi

if [ ! -s "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml is empty!"
    exit 1
fi

# Check Docker Compose syntax
if command -v docker-compose &>/dev/null; then
    echo "✅ Using docker-compose (standalone)"
    COMPOSE_CMD="docker-compose"
elif docker compose version &>/dev/null 2>&1; then
    echo "✅ Using docker compose (plugin)"
    COMPOSE_CMD="docker compose"
else
    echo "❌ Docker Compose not found!"
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
$COMPOSE_CMD down

# Build and start
echo "🔄 Building Pizza Sales Tracker..."
$COMPOSE_CMD build --no-cache

echo "🚀 Starting services..."
$COMPOSE_CMD up -d

# Check container status
echo "📊 Container status:"
$COMPOSE_CMD ps

# Wait and test
echo "⏳ Waiting for services to start..."
sleep 60

echo "🧪 Testing connection..."
if curl -f http://localhost:8501/_stcore/health 2>/dev/null; then
    echo "✅ Pizza Sales Tracker is ready!"
    echo "🍕 Access at: http://localhost:8501"
else
    echo "⚠️  Service not ready. Checking logs..."
    $COMPOSE_CMD logs pizza-robust-tracker
fi
