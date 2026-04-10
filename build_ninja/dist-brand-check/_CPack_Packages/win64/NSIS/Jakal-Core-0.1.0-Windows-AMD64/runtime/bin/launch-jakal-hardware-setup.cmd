@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0..\share\jakal-core\install\install-jakal-core.ps1" -InstallRoot "%~dp0.." %*
endlocal
