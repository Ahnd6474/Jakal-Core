param(
    [string]$BuildDir = "",
    [string]$ArchivePath = "",
    [string]$BuildConfig = "",
    [string]$WorkRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-PackageArchive {
    param(
        [string]$BuildDirectory,
        [string]$ExplicitArchivePath
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitArchivePath)) {
        return (Resolve-Path -Path $ExplicitArchivePath -ErrorAction Stop).Path
    }

    if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
        throw "Pass -ArchivePath or -BuildDir."
    }

    $resolvedBuildDir = Resolve-Path -Path $BuildDirectory -ErrorAction Stop
    $artifacts = @(
        Get-ChildItem -Path (Join-Path $resolvedBuildDir.Path "dist") -File -Filter *.zip -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTimeUtc -Descending
    )
    if ($artifacts.Count -eq 0) {
        return ""
    }
    return $artifacts[0].FullName
}

function New-SmokeRoot {
    param([string]$RequestedRoot)

    if (-not [string]::IsNullOrWhiteSpace($RequestedRoot)) {
        $item = New-Item -ItemType Directory -Force -Path $RequestedRoot
        return $item.FullName
    }

    $nonce = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $root = Join-Path ([System.IO.Path]::GetTempPath()) "jakal-package-smoke-$nonce"
    $item = New-Item -ItemType Directory -Force -Path $root
    return $item.FullName
}

function Assert-Exists {
    param([string]$PathToCheck, [string]$Message)

    if (-not (Test-Path -Path $PathToCheck)) {
        throw $Message
    }
}

$archive = Resolve-PackageArchive -BuildDirectory $BuildDir -ExplicitArchivePath $ArchivePath
$smokeRoot = New-SmokeRoot -RequestedRoot $WorkRoot
$extractRoot = Join-Path $smokeRoot "extracted"
$runtimeRoot = Join-Path $smokeRoot "runtime-home"
$updatePayloadRoot = Join-Path $smokeRoot "update-payload"
$updateTarget = Join-Path $smokeRoot "update-target"
$removeRoot = Join-Path $smokeRoot "remove-root"
$updateArchive = Join-Path $smokeRoot "update.zip"

New-Item -ItemType Directory -Force -Path $extractRoot, $runtimeRoot, $updatePayloadRoot, $updateTarget | Out-Null
if (-not [string]::IsNullOrWhiteSpace($archive)) {
    Expand-Archive -Path $archive -DestinationPath $extractRoot -Force

    $installRoot = Get-ChildItem -Path $extractRoot -Directory | Select-Object -First 1
    if ($null -eq $installRoot) {
        throw "Expanded archive does not contain a top-level install directory."
    }
    $installRoot = $installRoot.FullName
} else {
    if ([string]::IsNullOrWhiteSpace($BuildDir)) {
        throw "No package archive found and no build directory was provided."
    }
    $installRoot = Join-Path $smokeRoot "staged-install"
    $installArguments = @("--install", (Resolve-Path -Path $BuildDir -ErrorAction Stop).Path, "--prefix", $installRoot)
    if (-not [string]::IsNullOrWhiteSpace($BuildConfig)) {
        $installArguments += @("--config", $BuildConfig)
    }
    & cmake @installArguments
    if ($LASTEXITCODE -ne 0) {
        throw "cmake --install failed with exit code $LASTEXITCODE"
    }
}

$launchScript = Join-Path $installRoot "bin\\launch-jakal-hardware-setup.cmd"
$installScript = Join-Path $installRoot "share\\jakal-core\\install\\install-jakal-core.ps1"
$updateScript = Join-Path $installRoot "share\\jakal-core\\update\\update-jakal-core.ps1"
$updateHelper = Join-Path $installRoot "share\\jakal-core\\update\\sign-and-verify-artifact.ps1"
$installHelper = Join-Path $installRoot "share\\jakal-core\\install\\sign-and-verify-artifact.ps1"
$removeScript = Join-Path $installRoot "share\\jakal-core\\remove\\remove-jakal-core.ps1"
$runtimeDll = Join-Path $installRoot "bin\\jakal_runtime.dll"
$doctorCli = Join-Path $installRoot "bin\\jakal_core_cli.exe"

Assert-Exists -PathToCheck $launchScript -Message "Missing launch-jakal-hardware-setup.cmd in packaged bin/."
Assert-Exists -PathToCheck $installScript -Message "Missing installed install-jakal-core.ps1."
Assert-Exists -PathToCheck $updateScript -Message "Missing installed update-jakal-core.ps1."
Assert-Exists -PathToCheck $updateHelper -Message "Missing installed sign-and-verify-artifact.ps1 next to update helper."
Assert-Exists -PathToCheck $installHelper -Message "Missing installed sign-and-verify-artifact.ps1 next to install helper."
Assert-Exists -PathToCheck $removeScript -Message "Missing installed remove-jakal-core.ps1."
Assert-Exists -PathToCheck $runtimeDll -Message "Missing packaged jakal_runtime.dll."
Assert-Exists -PathToCheck $doctorCli -Message "Missing packaged jakal_core_cli.exe."

& $launchScript -RuntimeRoot $runtimeRoot -Backend cpu-only -Prerequisites none -Quiet
if ($LASTEXITCODE -ne 0) {
    throw "launch-jakal-hardware-setup.cmd failed with exit code $LASTEXITCODE"
}

$configPath = Join-Path $runtimeRoot "config\\jakal-runtime-config.ini"
Assert-Exists -PathToCheck $configPath -Message "Install helper did not write runtime config into overridden runtime root."
$configText = Get-Content -Path $configPath -Raw -Encoding ascii
if ($configText -notmatch 'host=1' -or $configText -notmatch 'level_zero=0') {
    throw "Install helper wrote an unexpected backend preset."
}

Set-Content -Path (Join-Path $updatePayloadRoot "sentinel.txt") -Value "package-smoke" -Encoding ascii
Compress-Archive -Path (Join-Path $updatePayloadRoot "*") -DestinationPath $updateArchive -Force
& $updateHelper -ArtifactPath $updateArchive -WriteChecksum -VerifyChecksum
if ($LASTEXITCODE -ne 0) {
    throw "sign-and-verify-artifact.ps1 failed to write/verify update checksum."
}

& $updateScript -InstallerPath $updateArchive -TargetDir $updateTarget -RequireChecksum -Quiet
if ($LASTEXITCODE -ne 0) {
    throw "update-jakal-core.ps1 failed with exit code $LASTEXITCODE"
}

$updatedFile = Join-Path $updateTarget "sentinel.txt"
Assert-Exists -PathToCheck $updatedFile -Message "Update helper did not extract ZIP payload."

Copy-Item -Path $installRoot -Destination $removeRoot -Recurse -Force
& $removeScript -InstallRoot $removeRoot -ForceRemove -Quiet
if ($LASTEXITCODE -ne 0) {
    throw "remove-jakal-core.ps1 failed with exit code $LASTEXITCODE"
}
if (Test-Path -Path $removeRoot) {
    throw "remove-jakal-core.ps1 did not remove the requested install root."
}

Write-Host "package smoke ok"
