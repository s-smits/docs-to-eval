#!/bin/bash

# docs-to-eval Server Startup Script
# This script properly starts the server with all necessary checks

set -e  # Exit on any error

echo "🚀 docs-to-eval Server Startup"
echo "==============================="

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider running: uv init && uv sync"
else
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
fi

# Check if .env file exists
if [[ -f ".env" ]]; then
    echo "✅ .env file found"
    
    # Check if API key is set
    if grep -q "OPENROUTER_API_KEY=sk-or-v1-" .env; then
        echo "✅ OPENROUTER_API_KEY appears to be configured"
    else
        echo "⚠️  OPENROUTER_API_KEY may not be configured properly"
        echo "   Check your .env file"
    fi
else
    echo "❌ .env file not found"
    echo "   Creating template .env file..."
    cat > .env << EOF
# docs-to-eval environment configuration
OPENROUTER_API_KEY=your_api_key_here
PYTORCH_ENABLE_MPS_FALLBACK=1
EOF
    echo "📝 Created .env file - please update with your API key"
fi

# Check if required directories exist
for dir in output logs cache uploads; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        echo "📁 Created directory: $dir"
    fi
done

# Install dependencies if needed
if [[ ! -f "pyproject.toml" ]]; then
    echo "❌ pyproject.toml not found - are you in the right directory?"
    exit 1
fi

# Check if packages are installed
echo "🔍 Checking dependencies..."
if ! python -c "import fastapi, uvicorn, websockets" 2>/dev/null; then
    echo "⚠️  Some dependencies missing - running uv sync..."
    uv sync
fi

echo "✅ All dependencies satisfied"

# Check if port 8080 is available
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 8080 is already in use"
    echo "   Kill existing process or use a different port"
    echo "   To kill: lsof -ti:8080 | xargs kill -9"
    
    read -p "Kill existing process and continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔪 Killing existing process..."
        lsof -ti:8080 | xargs kill -9 2>/dev/null || true
        sleep 2
    else
        echo "❌ Aborted"
        exit 1
    fi
fi

echo "🌐 Starting server on http://localhost:8080"
echo "📊 API documentation: http://localhost:8080/docs"
echo "🔧 Health check: http://localhost:8080/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "==============================="

# Start the server
python run_server.py