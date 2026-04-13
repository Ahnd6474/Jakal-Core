param(
    [string]$InstallRoot = "",
    [switch]$ForceRemove,
    [switch]$Quiet
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    $InstallRoot = Split-Path -Path $PSScriptRoot -Parent
    $InstallRoot = Split-Path -Path $InstallRoot -Parent
}

$nsisUninstaller = Join-Path $InstallRoot "Uninstall.exe"

if (Test-Path $nsisUninstaller) {
    $arguments = @()
    if ($Quiet) {
        $arguments += "/S"
    }
    Start-Process -FilePath $nsisUninstaller -ArgumentList $arguments -Wait
    exit 0
}

if (-not $ForceRemove) {
    throw "No packaged uninstaller found under $InstallRoot. Re-run with -ForceRemove to remove the install tree directly."
}

Remove-Item -Path $InstallRoot -Recurse -Force
