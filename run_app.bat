@echo off
setlocal
cd /d "%~dp0"
conda run --no-capture-output -n heartbeat python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
endlocal
