param(
  [string]$Message = "",
  [switch]$PullOnly
)

if ($Message -eq "" -and -not $PullOnly) {
  $Message = "Update " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
}

$repoRoot = git rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Error: Not a git repository"
  exit 1
}
Set-Location $repoRoot

Write-Host "`n=== GitHub Sync ===`n"

# Stage any local changes
git add -A

# Fetch from GitHub
Write-Host "Fetching from GitHub..."
git fetch origin 2>$null
$remoteExists = $LASTEXITCODE -eq 0

if ($remoteExists) {
  # Get commits
  $localCommit = git rev-parse HEAD 2>$null
  $remoteCommit = git rev-parse origin/main 2>$null
  
  # Merge if needed
  if ($localCommit -ne $remoteCommit) {
    Write-Host "Merging from GitHub..."
    
    git diff HEAD --quiet
    if ($LASTEXITCODE -ne 0) {
      git commit -m "Local changes before merge" 2>$null | Out-Null
    }
    
    git merge origin/main --no-edit 2>&1 | Out-Null
    
    if ($LASTEXITCODE -ne 0) {
      Write-Host "[!] Merge conflict. Keeping local version..."
      git merge --abort 2>$null
    } else {
      Write-Host "[OK] Merged from GitHub"
    }
  } else {
    Write-Host "[OK] Already up-to-date"
  }
}

if ($PullOnly) {
  Write-Host "`n[OK] Pull complete`n"
  exit 0
}

# Commit any remaining changes
git diff HEAD --quiet
if ($LASTEXITCODE -ne 0) {
  git commit -m $Message
  Write-Host "[OK] Committed: $Message"
}

# Push to GitHub
Write-Host "`nPushing to GitHub..."
git push origin main 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
  Write-Host "[OK] Push complete"
} else {
  git push origin main --force 2>&1 | Out-Null
  if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Push complete (forced)"
  } else {
    Write-Host "[ERROR] Push failed"
    exit 1
  }
}

Write-Host "`n[OK] Sync complete`n"
