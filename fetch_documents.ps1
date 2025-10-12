# ===============================================
# DOCUMENT FETCHER - POWERSHELL SCRIPT
# ===============================================

param(
    [string]$Cities = "",
    [switch]$ForceRefresh,
    [switch]$Verbose,
    [switch]$Stats,
    [switch]$Help
)

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "   CZECH MUNICIPAL DOCUMENT FETCHER" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

if ($Help) {
    Write-Host "Usage: .\fetch_documents.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor Green
    Write-Host "  -Cities <list>     Comma-separated city names (Praha,Brno,Ostrava)" -ForegroundColor White
    Write-Host "  -ForceRefresh      Force re-download cached data" -ForegroundColor White
    Write-Host "  -Verbose           Show detailed output" -ForegroundColor White
    Write-Host "  -Stats             Show collection statistics" -ForegroundColor White
    Write-Host "  -Help              Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\fetch_documents.ps1" -ForegroundColor Gray
    Write-Host "  .\fetch_documents.ps1 -Cities 'Praha,Brno' -Verbose" -ForegroundColor Gray
    Write-Host "  .\fetch_documents.ps1 -ForceRefresh -Stats" -ForegroundColor Gray
    Write-Host ""
    exit 0
}

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "  Please install Python 3.7+ and add it to PATH" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if required files exist
if (!(Test-Path "main.py")) {
    Write-Host "✗ ERROR: main.py not found in current directory" -ForegroundColor Red
    Write-Host "  Please run this script from the project root directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

if (!(Test-Path "fetch_documents.py")) {
    Write-Host "✗ ERROR: fetch_documents.py not found" -ForegroundColor Red
    Write-Host "  Please ensure the fetch_documents.py script exists" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Build command arguments
$arguments = @("fetch_documents.py")

if ($Cities) {
    $arguments += "--cities"
    $arguments += $Cities
}

if ($ForceRefresh) {
    $arguments += "--force-refresh"
}

if ($Verbose) {
    $arguments += "--verbose"
}

if ($Stats) {
    $arguments += "--output-stats"
}

# Display what we're doing
Write-Host "🔍 Configuration:" -ForegroundColor Blue
if ($Cities) {
    Write-Host "   Cities: $Cities" -ForegroundColor White
} else {
    Write-Host "   Cities: All supported (Děčín, Ústí nad Labem, Praha, Brno, Ostrava, Teplice)" -ForegroundColor White
}
Write-Host "   Force Refresh: $ForceRefresh" -ForegroundColor White
Write-Host "   Verbose Output: $Verbose" -ForegroundColor White
Write-Host "   Show Statistics: $Stats" -ForegroundColor White
Write-Host ""

# Execute the Python script
Write-Host "🚀 Starting document collection..." -ForegroundColor Green
Write-Host ""

try {
    & python $arguments
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Host ""
        Write-Host "===============================================" -ForegroundColor Cyan
        Write-Host "   DOCUMENT FETCHING COMPLETED SUCCESSFULLY" -ForegroundColor Green
        Write-Host "===============================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "📁 Documents are now available in:" -ForegroundColor Blue
        Write-Host "   • vector_store.json (AI search index)" -ForegroundColor White
        Write-Host "   • documents/ folder (raw files)" -ForegroundColor White
        Write-Host "   • nkod_cache/ folder (metadata)" -ForegroundColor White
        Write-Host ""
        Write-Host "🚀 Next steps:" -ForegroundColor Blue
        Write-Host "   • Start full system: .\start.bat" -ForegroundColor White
        Write-Host "   • Start API only: python api_server.py" -ForegroundColor White
        Write-Host "   • Test chatbot: python main.py" -ForegroundColor White
    } else {
        Write-Host ""
        Write-Host "⚠️  Document fetching completed with errors (exit code: $exitCode)" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host ""
    Write-Host "✗ ERROR: Failed to execute document fetcher" -ForegroundColor Red
    Write-Host "  $($_.Exception.Message)" -ForegroundColor Yellow
    $exitCode = 1
}

Write-Host ""
Read-Host "Press Enter to exit"
exit $exitCode