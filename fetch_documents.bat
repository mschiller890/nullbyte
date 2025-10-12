@echo off
REM ===============================================
REM   DOCUMENT FETCHER - BATCH SCRIPT
REM ===============================================
echo.
echo ===============================================
echo   CZECH MUNICIPAL DOCUMENT FETCHER
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "main.py" (
    echo ERROR: main.py not found in current directory
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

REM Default operation - fetch all cities
if "%1"=="" (
    echo [1] Fetching documents from all supported cities...
    python fetch_documents.py --verbose --output-stats
    goto :done
)

REM Handle command line arguments
if "%1"=="--help" (
    echo Usage: fetch_documents.bat [options]
    echo.
    echo Options:
    echo   --help          Show this help message
    echo   --refresh       Force refresh all cached data
    echo   --prague        Fetch only Prague documents
    echo   --brno          Fetch only Brno documents
    echo   --ostrava       Fetch only Ostrava documents
    echo   --all           Fetch all cities (default)
    echo   --stats         Show detailed statistics
    echo.
    pause
    exit /b 0
)

if "%1"=="--refresh" (
    echo [1] Force refreshing all documents...
    python fetch_documents.py --force-refresh --verbose --output-stats
    goto :done
)

if "%1"=="--prague" (
    echo [1] Fetching documents from Praha only...
    python fetch_documents.py --cities "Praha" --verbose --output-stats
    goto :done
)

if "%1"=="--brno" (
    echo [1] Fetching documents from Brno only...
    python fetch_documents.py --cities "Brno" --verbose --output-stats
    goto :done
)

if "%1"=="--ostrava" (
    echo [1] Fetching documents from Ostrava only...
    python fetch_documents.py --cities "Ostrava" --verbose --output-stats
    goto :done
)

if "%1"=="--all" (
    echo [1] Fetching documents from all cities...
    python fetch_documents.py --verbose --output-stats
    goto :done
)

if "%1"=="--stats" (
    echo [1] Quick fetch with detailed statistics...
    python fetch_documents.py --output-stats
    goto :done
)

REM Unknown argument
echo ERROR: Unknown argument "%1"
echo Use "fetch_documents.bat --help" for usage information
pause
exit /b 1

:done
echo.
echo ===============================================
echo   DOCUMENT FETCHING COMPLETED
echo ===============================================
echo.
echo The documents are now available for the chatbot.
echo You can start the full system with: start.bat
echo Or just the API server with: python api_server.py
echo.
pause