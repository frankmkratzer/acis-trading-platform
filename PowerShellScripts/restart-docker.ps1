# restart-docker.ps1
Clear-Host
Write-Host "🔄 Restarting Docker Desktop" -ForegroundColor Cyan

# Stop Docker Desktop
Write-Host "Stopping Docker Desktop..." -ForegroundColor Yellow
Stop-Process -Name "Docker Desktop" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "dockerd" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "com.docker.service" -Force -ErrorAction SilentlyContinue

Write-Host "Waiting 5 seconds..." -ForegroundColor Gray
Start-Sleep -Seconds 5

# Start Docker Desktop
Write-Host "Starting Docker Desktop..." -ForegroundColor Yellow
$dockerPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
if (Test-Path $dockerPath) {
    Start-Process $dockerPath
    Write-Host "✅ Docker Desktop started" -ForegroundColor Green

    Write-Host "`nWaiting for Docker daemon to be ready (this may take 30-60 seconds)..." -ForegroundColor Yellow

    # Wait with progress indicator
    $dockerExe = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
    $maxWait = 60
    $waited = 0

    while ($waited -lt $maxWait) {
        Write-Host -NoNewline "`r⏱️ Waited $waited seconds... " -ForegroundColor Gray

        & $dockerExe version 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "✅ Docker is ready!" -ForegroundColor Green

            # Now run your services
            Write-Host "`nStarting ACIS Trading Platform..." -ForegroundColor Cyan
            Set-Location "C:\Users\frank\PycharmProjects\PythonProject\acis-trading-platform"

            # Add to PATH
            $env:PATH = "C:\Program Files\Docker\Docker\resources\bin;$env:PATH"

            docker-compose up -d
            docker-compose ps
            break
        }

        Start-Sleep -Seconds 2
        $waited += 2
    }

    if ($waited -ge $maxWait) {
        Write-Host ""
        Write-Host "❌ Docker daemon didn't start after $maxWait seconds" -ForegroundColor Red
        Write-Host "Please check Docker Desktop settings" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ Docker Desktop not found at expected location" -ForegroundColor Red
}

Read-Host "`nPress Enter to exit"