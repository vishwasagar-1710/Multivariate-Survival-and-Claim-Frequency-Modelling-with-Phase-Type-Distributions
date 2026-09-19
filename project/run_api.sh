#!/usr/bin/env bash
# Launch the risk-scoring API. Run from the project root:
#   bash run_api.sh
set -e
uvicorn api.main:app --host 0.0.0.0 --port 8000
