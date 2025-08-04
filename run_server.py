#!/usr/bin/env python3
"""
Simple server runner to avoid multiprocessing issues
"""

import uvicorn
from docs_to_eval.ui_api.main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="localhost",
        port=8080,
        log_level="info",
        reload=False  # Disable reload to avoid multiprocessing issues
    )