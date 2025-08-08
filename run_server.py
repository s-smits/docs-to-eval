#!/usr/bin/env python3
"""
Simple server runner to avoid multiprocessing issues
"""

import os
import sys
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"✅ Loaded environment variables from {env_path}")
        
        # Verify critical environment variables
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            print(f"✅ OPENROUTER_API_KEY loaded (length: {len(api_key)})")
        else:
            print("⚠️  OPENROUTER_API_KEY not found in .env file")
            
        pytorch_fallback = os.getenv("PYTORCH_ENABLE_MPS_FALLBACK")
        if pytorch_fallback:
            print(f"✅ PYTORCH_ENABLE_MPS_FALLBACK: {pytorch_fallback}")
            
        # Check testing mode
        testing_mode = os.getenv("DOCS_TO_EVAL_TESTING_MODE", "false").lower()
        if testing_mode in ["true", "1", "yes"]:
            print("🧪 TESTING MODE ENABLED - Full agentic loops will run every time")
        else:
            print("📊 Production mode - Normal chunking behavior")
    else:
        print(f"⚠️  Environment file not found: {env_path}")
        print("Creating basic .env file with placeholders...")
        
        with open(env_path, 'w') as f:
            f.write("# docs-to-eval environment configuration\n")
            f.write("OPENROUTER_API_KEY=your_api_key_here\n")
            f.write("PYTORCH_ENABLE_MPS_FALLBACK=1\n")
            f.write("# Set to true to disable caching and force full agentic loops\n")
            f.write("DOCS_TO_EVAL_TESTING_MODE=false\n")
        
        print(f"📝 Created {env_path} - please update with your API key")

if __name__ == "__main__":
    print("🚀 Starting docs-to-eval server...")
    
    # Load environment first
    load_environment()
    
    # Import app after environment is loaded
    try:
        from docs_to_eval.ui_api.main import app
        print("✅ Successfully imported application")
    except ImportError as e:
        print(f"❌ Failed to import application: {e}")
        sys.exit(1)
    
    # Start server
    print("🌐 Starting Uvicorn server on http://localhost:8080")
    uvicorn.run(
        app,
        host="localhost",
        port=8080,
        log_level="info",
        reload=False,  # Disable reload to avoid multiprocessing issues
        access_log=True
    )