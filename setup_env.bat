@echo off
setlocal
cd /d "%~dp0"
conda env update -n heartbeat -f environment.yml --prune
if errorlevel 1 (
  echo Failed to update existing heartbeat environment. Trying to create it...
  conda env create -f environment.yml
)
endlocal
