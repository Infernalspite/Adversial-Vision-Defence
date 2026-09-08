$ErrorActionPreference = 'Stop'
npm --prefix "$PSScriptRoot\..\frontend" run dev -- --host 0.0.0.0
