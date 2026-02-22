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

# ── STEP 1: Auto-generate index.json by scanning all folders ─────────────────
Write-Host "Scanning folders and generating index.json..."

$pythonScript = @"
import os, re, json

ROOT = r'$repoRoot'

# Folders/files to ignore at any level
SKIP_FOLDERS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'debug', '.github'}
SKIP_FILES   = {'index.html', 'index.json', 'sync.ps1', 'readme.md', '404.html'}

def normalize(name):
    return re.sub(r'^\d+\.\s*', '', name).strip()

def sort_key(name):
    m = re.match(r'^(\d+)', name)
    return (int(m.group(1)), name) if m else (999999, name)

def safe_listdir(path):
    try:    return sorted(os.listdir(path), key=sort_key)
    except: return []

def collect_html(folder_path, rel_prefix):
    files = []
    for fn in safe_listdir(folder_path):
        if fn.lower() in SKIP_FILES:     continue
        if not fn.endswith('.html'):      continue
        full = os.path.join(folder_path, fn)
        if not os.path.isfile(full):      continue
        files.append({
            'name': normalize(os.path.splitext(fn)[0]),
            'html': f'{rel_prefix}/{fn}'
        })
    return files

structure = []

for folder_name in safe_listdir(ROOT):
    if folder_name in SKIP_FOLDERS:       continue
    folder_path = os.path.join(ROOT, folder_name)
    if not os.path.isdir(folder_path):    continue
    if os.path.islink(folder_path):       continue

    rel = folder_name   # path relative to repo root

    # Collect direct HTML files in this folder
    direct_files = collect_html(folder_path, rel)

    # Collect subfolders that contain HTML files
    subfolders = []
    for sub_name in safe_listdir(folder_path):
        if sub_name in SKIP_FOLDERS:      continue
        sub_path = os.path.join(folder_path, sub_name)
        if not os.path.isdir(sub_path):   continue
        if os.path.islink(sub_path):      continue
        sub_files = collect_html(sub_path, f'{rel}/{sub_name}')
        if sub_files:
            subfolders.append({'name': normalize(sub_name), 'folder': sub_name, 'files': sub_files})

    # Only add folder if it has something to show
    if direct_files or subfolders:
        structure.append({
            'name':       normalize(folder_name),
            'folder':     folder_name,
            'files':      direct_files,
            'subfolders': subfolders
        })

out = os.path.join(ROOT, 'index.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(structure, f, indent=2, ensure_ascii=False)

total = sum(len(s['files']) + sum(len(sf['files']) for sf in s['subfolders']) for s in structure)
print(f'index.json -> {len(structure)} folder(s), {total} file(s)')
"@

python -c $pythonScript

if ($LASTEXITCODE -ne 0) {
  Write-Host "[ERROR] index.json generation failed — aborting"
  exit 1
}

# ── STEP 2: Stage everything ──────────────────────────────────────────────────
git add -A

# ── STEP 3: Fetch + merge from GitHub ────────────────────────────────────────
Write-Host "`nFetching from GitHub..."
git fetch origin 2>$null
$remoteExists = $LASTEXITCODE -eq 0

if ($remoteExists) {
  $localCommit  = git rev-parse HEAD 2>$null
  $remoteCommit = git rev-parse origin/main 2>$null

  if ($localCommit -ne $remoteCommit) {
    Write-Host "Merging from GitHub..."

    git diff HEAD --quiet
    if ($LASTEXITCODE -ne 0) {
      git commit -m "Local changes before merge" 2>$null | Out-Null
    }

    git merge origin/main --no-edit 2>&1 | Out-Null

    if ($LASTEXITCODE -ne 0) {
      Write-Host "[!] Merge conflict — keeping local version..."
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

# ── STEP 4: Commit ────────────────────────────────────────────────────────────
git diff HEAD --quiet
if ($LASTEXITCODE -ne 0) {
  git commit -m $Message
  Write-Host "[OK] Committed: $Message"
} else {
  Write-Host "[OK] Nothing new to commit"
}

# ── STEP 5: Push ──────────────────────────────────────────────────────────────
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