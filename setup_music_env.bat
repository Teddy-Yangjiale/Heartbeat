@echo off
setlocal
cd /d "%~dp0"

conda env list | findstr /R /C:"^heartbeat-music " >nul
if errorlevel 1 (
  conda env create -f environment-music.yml
) else (
  conda env update -n heartbeat-music -f environment-music.yml --prune
)
if errorlevel 1 exit /b 1

set "MODEL_DIR=%USERPROFILE%\.cache\torch\hub\checkpoints"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if not exist "%MODEL_DIR%\955717e8-8726e21a.th" (
  curl.exe -L --fail --retry 3 -o "%MODEL_DIR%\955717e8-8726e21a.th" "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
)
if errorlevel 1 exit /b 1

conda run --no-capture-output -n heartbeat-music python -c "import BeatNet, demucs, basic_pitch; print('heartbeat-music model packages are available')"
endlocal
