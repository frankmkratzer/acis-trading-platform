# check-docker-status.ps1
Clear-Host
Write-Host "🔍 Docker Daemon Diagnostic" -ForegroundColor Cyan
Write-Host "=" * 50

# Check Docker Desktop process
Write-Host "`n1. Docker Desktop Process:" -ForegroundColor Yellow
$dockerDesktop = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
if ($dockerDesktop) {
    Write-Host "✅ Docker Desktop is running (PID: $($dockerDesktop.Id))" -ForegroundColor Green
    Write-Host "   Memory: $([math]::Round($dockerDesktop.WorkingSet64 / 1MB, 2)) MB" -ForegroundColor Gray
}

# Check Docker daemon process
Write-Host "`n2. Docker Daemon Process:" -ForegroundColor Yellow
$dockerd = Get-Process "dockerd" -ErrorAction SilentlyContinue
if ($dockerd) {
    Write-Host "✅ Docker daemon is running (PID: $($dockerd.Id))" -ForegroundColor Green
} else {
    Write-Host "⚠️ Docker daemon process not found" -ForegroundColor Yellow
}

# Check com.docker.service
Write-Host "`n3. Docker Service:" -ForegroundColor Yellow
$dockerService = Get-Process "com.docker.service" -ErrorAction SilentlyContinue
if ($dockerService) {
    Write-Host "✅ Docker service is running" -ForegroundColor Green
} else {
    Write-Host "⚠️ Docker service not found" -ForegroundColor Yellow
}

# Check WSL2 backend
Write-Host "`n4. WSL2 Backend:" -ForegroundColor Yellow
$wslProcess = Get-Process "wsl" -ErrorAction SilentlyContinue
if ($wslProcess) {
    Write-Host "✅ WSL processes found" -ForegroundColor Green
} else {
    Write-Host "⚠️ No WSL processes (Docker might be using Hyper-V)" -ForegroundColor Yellow
}

# Check if Hyper-V is running
Write-Host "`n5. Virtualization:" -ForegroundColor Yellow
$hyperv = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All -ErrorAction SilentlyContinue
if ($hyperv -and $hyperv.State -eq "Enabled") {
    Write-Host "✅ Hyper-V is enabled" -ForegroundColor Green
} else {
    # Check for WSL2
    $wslStatus = wsl --status 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ WSL2 is available" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Check virtualization settings" -ForegroundColor Yellow
    }
}

# Check Docker settings file
Write-Host "`n6. Docker Settings:" -ForegroundColor Yellow
$settingsPath = "$env:APPDATA\Docker\settings.json"
if (Test-Path $settingsPath) {
    Write-Host "✅ Settings file exists" -ForegroundColor Green
    $settings = Get-Content $settingsPath | ConvertFrom-Json
    Write-Host "   WSL2 Enabled: $($settings.wslEngineEnabled)" -ForegroundColor Gray
} else {
    Write-Host "⚠️ Settings file not found" -ForegroundColor Yellow
}

# Try to connect with timeout
Write-Host "`n7. Testing Docker Connection:" -ForegroundColor Yellow
Write-Host "   Attempting connection..." -ForegroundColor Gray

# Direct test using full path
$dockerExe = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
$timeout = 10
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

while ($stopwatch.Elapsed.TotalSeconds -lt $timeout) {
    & $dockerExe version 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker is responding!" -ForegroundColor Green
        $stopwatch.Stop()

        # Show version
        Write-Host "`nDocker Version:" -ForegroundColor Yellow
        & $dockerExe version
        break
    }
    Start-Sleep -Milliseconds 500
    Write-Host -NoNewline "."
}

if ($stopwatch.Elapsed.TotalSeconds -ge $timeout) {
    Write-Host ""
    Write-Host "❌ Docker not responding after $timeout seconds" -ForegroundColor Red
}

Write-Host "`n"
Read-Host "Press Enter to continue"