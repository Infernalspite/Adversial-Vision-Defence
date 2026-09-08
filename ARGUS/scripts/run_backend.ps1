$ErrorActionPreference = 'Stop'
Set-Location "$PSScriptRoot\..\backend"
if (Test-Path '.venv\Scripts\uvicorn.exe') { & '.venv\Scripts\uvicorn.exe' app.main:app --reload --host 0.0.0.0 --port 8000 } else { uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 }
