@echo off
setlocal
cd /d "%~dp0"
echo Starting on all network interfaces. Allow Windows Firewall if prompted.
conda run --no-capture-output -n heartbeat python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false
endlocal
