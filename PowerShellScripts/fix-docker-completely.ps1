# fix-docker-completely.ps1
Clear-Host
Write-Host "DOCKER DESKTOP COMPLETE FIX" -ForegroundColor Cyan
Write-Host ("=" * 50)

# Run as Administrator check
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "WARNING: Please run this script as Administrator!" -ForegroundColor Yellow
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit
}

Write-Host "OK: Running as Administrator" -ForegroundColor Green

# Step 1: Stop all Docker processes
Write-Host "`n[1/6] Stopping all Docker processes..." -ForegroundColor Yellow
$processes = @("Docker Desktop", "docker", "dockerd", "com.docker.service", "vpnkit", "com.docker.backend")
foreach ($proc in $processes) {
    Stop-Process -Name $proc -Force -ErrorAction SilentlyContinue
}
Write-Host "OK: Docker processes stopped" -ForegroundColor Green

# Step 2: Check WSL2
Write-Host "`n[2/6] Checking WSL2..." -ForegroundColor Yellow
$wslVersion = wsl --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK: WSL2 is installed" -ForegroundColor Green

    # Update WSL
    Write-Host "Updating WSL..." -ForegroundColor Gray
    wsl --update

    # Shutdown WSL
    wsl --shutdown
    Write-Host "OK: WSL updated and reset" -ForegroundColor Green
} else {
    Write-Host "WARNING: WSL2 not properly installed" -ForegroundColor Yellow
    Write-Host "Installing WSL2..." -ForegroundColor Cyan
    wsl --install --no-distribution
}

# Step 3: Check Hyper-V
Write-Host "`n[3/6] Checking virtualization..." -ForegroundColor Yellow
$hyperv = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All
if ($hyperv.State -eq "Enabled") {
    Write-Host "OK: Hyper-V is enabled" -ForegroundColor Green
} else {
    Write-Host "WARNING: Hyper-V is not enabled" -ForegroundColor Yellow
    Write-Host "Enabling Hyper-V (requires restart)..." -ForegroundColor Cyan
    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All -NoRestart
}

# Step 4: Reset Docker Desktop data
Write-Host "`n[4/6] Resetting Docker Desktop settings..." -ForegroundColor Yellow
$dockerDataPath = "$env:APPDATA\Docker"
if (Test-Path $dockerDataPath) {
    $settingsFile = "$dockerDataPath\settings.json"
    if (Test-Path $settingsFile) {
        # Backup settings
        Copy-Item $settingsFile "$settingsFile.backup" -Force
        Write-Host "OK: Settings backed up" -ForegroundColor Green

        # Reset to WSL2 backend
        $settings = Get-Content $settingsFile | ConvertFrom-Json
        $settings.wslEngineEnabled = $true
        $settings | ConvertTo-Json -Depth 10 | Set-Content $settingsFile
        Write-Host "OK: Settings reset to use WSL2" -ForegroundColor Green
    }
}

# Step 5: Clear Docker cache
Write-Host "`n[5/6] Clearing Docker cache..." -ForegroundColor Yellow
$cachePaths = @(
    "$env:LOCALAPPDATA\Docker",
    "$env:ProgramData\Docker"
)

foreach ($path in $cachePaths) {
    if (Test-Path "$path\log") {
        Remove-Item "$path\log\*" -Force -Recurse -ErrorAction SilentlyContinue
        Write-Host "OK: Cleared logs in $path" -ForegroundColor Green
    }
}

# Step 6: Start Docker Desktop
Write-Host "`n[6/6] Starting Docker Desktop..." -ForegroundColor Yellow
$dockerExePath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"

if (Test-Path $dockerExePath) {
    Start-Process $dockerExePath
    Write-Host "OK: Docker Desktop launched" -ForegroundColor Green

    Write-Host "`nWaiting for Docker to initialize (this may take 60-90 seconds)..." -ForegroundColor Yellow
    Write-Host "   Docker Desktop window should appear soon" -ForegroundColor Gray

    # Wait for Docker
    $maxWait = 90
    $waited = 0
    $dockerReady = $false

    while ($waited -lt $maxWait -and -not $dockerReady) {
        Start-Sleep -Seconds 5
        $waited += 5

        Write-Host "Checking... $waited/$maxWait seconds" -ForegroundColor Gray

        # Check if Docker daemon is responding
        $dockerExe = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
        & $dockerExe version 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $dockerReady = $true
            Write-Host "OK: Docker is responding!" -ForegroundColor Green
        }
    }

    if (-not $dockerReady) {
        Write-Host "WARNING: Docker is taking longer than expected" -ForegroundColor Yellow
        Write-Host "Please wait for Docker Desktop to fully start" -ForegroundColor Yellow
        Write-Host "Look for the Docker whale icon in the system tray" -ForegroundColor Yellow
    }
} else {
    Write-Host "ERROR: Docker Desktop not found!" -ForegroundColor Red
    Write-Host "Please reinstall from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
}

Write-Host "`n$('=' * 50)" -ForegroundColor Cyan
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "1. Wait for Docker Desktop window to appear" -ForegroundColor White
Write-Host "2. If prompted, accept the Docker Service Agreement" -ForegroundColor White
Write-Host "3. Look for the whale icon in system tray (should stop animating)" -ForegroundColor White
Write-Host "4. Once ready, run your test script again" -ForegroundColor White

Read-Host "`nPress Enter to exit"