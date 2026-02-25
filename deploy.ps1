param(
    [string]$CommitMessage = "Update attendance system security and UI"
)

# 1. Git Automation
Write-Host ">>> Starting Git automation..." -ForegroundColor Cyan

# Check if there are changes to commit
$gitStatus = git status --porcelain
if ($null -ne $gitStatus -and $gitStatus -ne "") {
    Write-Host "Staging changes..."
    git add .
    
    Write-Host "Committing changes with message: '$CommitMessage'..."
    git commit -m $CommitMessage
    
    # Optional: push to origin if configured
    $remote = git remote
    if ($null -ne $remote -and $remote -ne "") {
        Write-Host "Pushing to origin..."
        git push origin main
    }
} else {
    Write-Host "No changes to commit." -ForegroundColor Yellow
}

# 2. Docker Automation
Write-Host "`n>>> Starting Docker build and restart..." -ForegroundColor Cyan

# Rebuild and restart the 'web' service
# Using --build to force a rebuild and -d to run in detached mode
docker-compose up -d --build web

Write-Host "`n>>> Automation complete!" -ForegroundColor Green
Write-Host "Service is now running at: http://localhost:5000"
