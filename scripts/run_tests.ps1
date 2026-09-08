$ErrorActionPreference = 'Stop'
Push-Location "$PSScriptRoot\..\backend"
pytest
Pop-Location
Push-Location "$PSScriptRoot\..\frontend"
npm run test:run
Pop-Location
