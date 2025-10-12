@echo off
echo ===============================================
echo   UREDNIDESKA CHATBOT - STARTUP SCRIPT
echo ===============================================
echo.

echo [1] Checking Python and installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)

echo.
echo [2] Starting Flask API server...
start "API Server" cmd /k "python api_server.py"

echo.
echo [3] Installing Node.js dependencies...
cd frontend
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Node.js dependencies
    pause
    exit /b 1
)

echo.
echo [4] Starting React frontend...
start "React Frontend" cmd /k "npm start"

echo.
echo ===============================================
echo   STARTUP COMPLETE!
echo ===============================================
echo.
echo Frontend: http://localhost:3000
echo API:      http://localhost:5000
echo.
echo Make sure Ollama is running: ollama serve
echo And model is available: ollama pull gemma3:4b
echo.
pause