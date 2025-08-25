# check-wsl.ps1
Write-Host "Checking for WSL..." -ForegroundColor Cyan

# Check if WSL is installed
$wslInstalled = Get-Command wsl -ErrorAction SilentlyContinue
if ($wslInstalled) {
    Write-Host "✅ WSL is installed" -ForegroundColor Green

    # Check WSL version
    wsl --list --verbose

    # Try docker through WSL
    Write-Host "`nTrying Docker through WSL..." -ForegroundColor Yellow
    wsl docker version

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker works through WSL!" -ForegroundColor Green
        Write-Host "`nYou can run Docker commands using: wsl docker [command]" -ForegroundColor Cyan
        Write-Host "Example: wsl docker-compose up -d" -ForegroundColor Gray
    }
} else {
    Write-Host "❌ WSL not installed" -ForegroundColor Red
}