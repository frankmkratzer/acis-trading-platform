# docker-setup-and-test.ps1
# Complete Docker setup and test for ACIS Trading Platform

Clear-Host
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     DOCKER SETUP & TEST UTILITY       ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan

# Function to find Docker
function Find-Docker {
    $locations = @(
        "C:\Program Files\Docker\Docker\resources\bin",
        "C:\ProgramData\DockerDesktop\version-bin",
        "$env:ProgramFiles\Docker\Docker\resources\bin",
        "$env:LOCALAPPDATA\Docker\bin"
    )

    foreach ($path in $locations) {
        if (Test-Path "$path\docker.exe") {
            return $path
        }
    }
    return $null
}

# Step 1: Locate Docker
Write-Host "`n[Step 1] Locating Docker..." -ForegroundColor Yellow
$dockerPath = Find-Docker

if ($dockerPath) {
    Write-Host "✅ Found Docker at: $dockerPath" -ForegroundColor Green

    # Add to PATH for current session
    $env:PATH = "$dockerPath;$env:PATH"
    Write-Host "✅ Added to current session PATH" -ForegroundColor Green
} else {
    Write-Host "❌ Docker executable not found!" -ForegroundColor Red
    Write-Host "Please ensure Docker Desktop is installed" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit
}

# Step 2: Verify Docker Desktop is running
Write-Host "`n[Step 2] Checking Docker Desktop..." -ForegroundColor Yellow
$dockerProcess = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue

if (-not $dockerProcess) {
    Write-Host "⚠️ Docker Desktop is not running!" -ForegroundColor Yellow
    Write-Host "Starting Docker Desktop..." -ForegroundColor Cyan

    $dockerDesktopPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dockerDesktopPath) {
        Start-Process $dockerDesktopPath
        Write-Host "Waiting 30 seconds for Docker to start..." -ForegroundColor Yellow

        # Show countdown
        for ($i = 30; $i -gt 0; $i--) {
            Write-Host -NoNewline "`r$i seconds remaining... " -ForegroundColor Gray
            Start-Sleep -Seconds 1
        }
        Write-Host ""
    }
} else {
    Write-Host "✅ Docker Desktop is running" -ForegroundColor Green
}

# Step 3: Test Docker command
Write-Host "`n[Step 3] Testing Docker CLI..." -ForegroundColor Yellow
$testDocker = docker version 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Docker CLI is working!" -ForegroundColor Green
} else {
    Write-Host "⚠️ Docker CLI not responding, waiting for Docker daemon..." -ForegroundColor Yellow

    # Wait for Docker daemon
    $attempts = 0
    $maxAttempts = 30

    while ($attempts -lt $maxAttempts) {
        $attempts++
        Write-Host -NoNewline "`rAttempt $attempts/$maxAttempts... " -ForegroundColor Gray

        docker version > $null 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "✅ Docker is now responding!" -ForegroundColor Green
            break
        }
        Start-Sleep -Seconds 2
    }

    if ($attempts -eq $maxAttempts) {
        Write-Host ""
        Write-Host "❌ Docker daemon not responding after $maxAttempts attempts" -ForegroundColor Red
        Write-Host "Please check Docker Desktop in system tray" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit
    }
}

# Step 4: Test docker-compose
Write-Host "`n[Step 4] Testing Docker Compose..." -ForegroundColor Yellow

# Try docker-compose first
docker-compose version > $null 2>&1
$composeWorks = $false
$composeCommand = ""

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 'docker-compose' command works" -ForegroundColor Green
    $composeWorks = $true
    $composeCommand = "docker-compose"
} else {
    # Try new syntax
    docker compose version > $null 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ 'docker compose' command works (new syntax)" -ForegroundColor Green
        $composeWorks = $true
        $composeCommand = "docker compose"
    }
}

if (-not $composeWorks) {
    Write-Host "❌ Docker Compose not working" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit
}

# Step 5: Navigate to project and start services
Write-Host "`n[Step 5] Starting ACIS Trading Platform..." -ForegroundColor Yellow

$projectPath = "C:\Users\frank\PycharmProjects\PythonProject\acis-trading-platform"
if (Test-Path $projectPath) {
    Set-Location $projectPath
    Write-Host "📁 Changed to project directory" -ForegroundColor Green
} else {
    Write-Host "❌ Project directory not found: $projectPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit
}

# Check for docker-compose.yml
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "❌ docker-compose.yml not found!" -ForegroundColor Red
    Write-Host "Files in current directory:" -ForegroundColor Yellow
    Get-ChildItem | Select-Object Name
    Read-Host "Press Enter to exit"
    exit
}

Write-Host "✅ Found docker-compose.yml" -ForegroundColor Green

# Start services
Write-Host "`nStarting services with: $composeCommand up -d" -ForegroundColor Cyan
Invoke-Expression "$composeCommand up -d"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Services started successfully!" -ForegroundColor Green
    Write-Host "Waiting 10 seconds for initialization..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
} else {
    Write-Host "❌ Failed to start services" -ForegroundColor Red
    Write-Host "`nShowing last 20 lines of logs:" -ForegroundColor Yellow
    Invoke-Expression "$composeCommand logs --tail=20"
}

# Step 6: Check service status
Write-Host "`n[Step 6] Service Status:" -ForegroundColor Yellow
Invoke-Expression "$composeCommand ps"

# Step 7: Test endpoints
Write-Host "`n[Step 7] Testing Endpoints:" -ForegroundColor Yellow

$tests = @(
    @{Name="PostgreSQL Port"; Type="Port"; Port=5432},
    @{Name="Redis Port"; Type="Port"; Port=6379},
    @{Name="API Health"; Type="HTTP"; Url="http://localhost/health"},
    @{Name="API Direct"; Type="HTTP"; Url="http://localhost:8000"},
    @{Name="Grafana"; Type="HTTP"; Url="http://localhost:3000"},
    @{Name="Prometheus"; Type="HTTP"; Url="http://localhost:9090"}
)

$passed = 0
$total = 0

foreach ($test in $tests) {
    $total++
    if ($test.Type -eq "Port") {
        $connection = Test-NetConnection -ComputerName localhost -Port $test.Port -WarningAction SilentlyContinue -InformationLevel Quiet
        if ($connection) {
            Write-Host "✅ $($test.Name) (Port $($test.Port)): OPEN" -ForegroundColor Green
            $passed++
        } else {
            Write-Host "❌ $($test.Name) (Port $($test.Port)): CLOSED" -ForegroundColor Red
        }
    } else {
        try {
            $response = Invoke-WebRequest -Uri $test.Url -TimeoutSec 3 -ErrorAction Stop
            Write-Host "✅ $($test.Name): OK (Status $($response.StatusCode))" -ForegroundColor Green
            $passed++
        } catch {
            Write-Host "⚠️ $($test.Name): Not responding" -ForegroundColor Yellow
        }
    }
}

# Summary
Write-Host "`n════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "📊 SUMMARY" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Tests Passed: $passed/$total" -ForegroundColor $(if ($passed -eq $total) {"Green"} elseif ($passed -gt 0) {"Yellow"} else {"Red"})

# Provide helpful commands
Write-Host "`n📝 Useful Commands:" -ForegroundColor Cyan
Write-Host "  View logs:     $composeCommand logs -f [service]" -ForegroundColor Gray
Write-Host "  Stop all:      $composeCommand down" -ForegroundColor Gray
Write-Host "  Clean restart: $composeCommand down -v && $composeCommand up -d" -ForegroundColor Gray

# Save the working configuration
$configFile = "docker-config.txt"
@"
Docker Path: $dockerPath
Compose Command: $composeCommand
Project Path: $projectPath
Generated: $(Get-Date)
"@ | Out-File $configFile

Write-Host "`n✅ Configuration saved to docker-config.txt" -ForegroundColor Green

Write-Host "`n"
Read-Host "Press Enter to exit"