import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure the root directory is in the path for module resolution
import sys
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

if __name__ == "__main__":
    # You can customize host and port here
    print("🚀 Starting RAG Hallucination Detector...")
    uvicorn.run(
        "api.main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True
    )
