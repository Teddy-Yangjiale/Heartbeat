@echo off
cd /d "%~dp0"
conda run --no-capture-output -n heartbeat python -m uvicorn studio_server:app --host 127.0.0.1 --port 8503
