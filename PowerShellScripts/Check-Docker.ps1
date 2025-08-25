# Check-Docker.ps1
Write-Host "`n🔍 Checking Docker Installation..." -ForegroundColor Cyan

# Check for Docker Desktop installation
$dockerDesktopPath = @(
    "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
    "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe",
    "$env:LOCALAPPDATA\Docker\Docker Desktop.exe"
)

$dockerFound = $false
foreach ($path in $dockerDesktopPath) {
    if (Test-Path $path) {
        Write-Host "✅ Docker Desktop found at: $path" -ForegroundColor Green
        $dockerFound = $true
        break
    }
}

if (-not $dockerFound) {
    Write-Host "❌ Docker Desktop not found!" -ForegroundColor Red
    Write-Host "`nPlease install Docker Desktop from:" -ForegroundColor Yellow
    Write-Host "https://www.docker.com/products/docker-desktop/" -ForegroundColor Cyan
    Read-Host "`nPress Enter to exit"
    exit
}

# Check if Docker is running
$dockerProcess = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
if ($dockerProcess) {
    Write-Host "✅ Docker Desktop is running" -ForegroundColor Green
} else {
    Write-Host "⚠️ Docker Desktop is installed but not running" -ForegroundColor Yellow
    Write-Host "Please start Docker Desktop from your Start Menu" -ForegroundColor Yellow

    # Try to start Docker Desktop
    $startDocker = Read-Host "`nWould you like to start Docker Desktop now? (Y/N)"
    if ($startDocker -eq 'Y') {
        Start-Process "Docker Desktop"
        Write-Host "Starting Docker Desktop... Please wait 30-60 seconds for it to initialize" -ForegroundColor Yellow
        Start-Sleep -Seconds 30
    }
}

# Check if docker command is available
Write-Host "`n🔍 Checking Docker CLI..." -ForegroundColor Cyan
try {
    $dockerVersion = & "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe" version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker CLI is working" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️ Docker CLI not in PATH" -ForegroundColor Yellow

    # Add to PATH for current session
    $dockerPath = "$env:ProgramFiles\Docker\Docker\resources\bin"
    if (Test-Path $dockerPath) {
        $env:PATH = "$dockerPath;$env:PATH"
        Write-Host "Added Docker to PATH for this session" -ForegroundColor Green
    }
}

# Check docker-compose
Write-Host "`n🔍 Checking Docker Compose..." -ForegroundColor Cyan
$composeCommands = @("docker-compose", "docker compose")
$composeWorks = $false

foreach ($cmd in $composeCommands) {
    try {
        $null = Invoke-Expression "$cmd version" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ '$cmd' command is working" -ForegroundColor Green
            $composeWorks = $true
            break
        }
    } catch {}
}

if (-not $composeWorks) {
    Write-Host "⚠️ Docker Compose not found" -ForegroundColor Yellow
    Write-Host "Using 'docker compose' (new syntax) instead" -ForegroundColor Yellow
}

Write-Host "`n✅ Setup check complete!" -ForegroundColor Green
Read-Host "`nPress Enter to continue"
