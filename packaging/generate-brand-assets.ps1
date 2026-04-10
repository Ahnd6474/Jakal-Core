param(
    [Parameter(Mandatory = $true)]
    [string]$SourcePng,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [string]$BaseName = "jakal-core-logo"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

function New-SquareBitmap {
    param(
        [Parameter(Mandatory = $true)]
        [System.Drawing.Image]$Source,
        [Parameter(Mandatory = $true)]
        [int]$Size
    )

    $bitmap = New-Object System.Drawing.Bitmap($Size, $Size, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
        $graphics.Clear([System.Drawing.Color]::Transparent)

        $scale = [Math]::Min($Size / [double]$Source.Width, $Size / [double]$Source.Height)
        $drawWidth = [int][Math]::Round($Source.Width * $scale)
        $drawHeight = [int][Math]::Round($Source.Height * $scale)
        $offsetX = [int][Math]::Floor(($Size - $drawWidth) / 2.0)
        $offsetY = [int][Math]::Floor(($Size - $drawHeight) / 2.0)
        $targetRect = New-Object System.Drawing.Rectangle($offsetX, $offsetY, $drawWidth, $drawHeight)
        $graphics.DrawImage($Source, $targetRect)

        return $bitmap
    }
    finally {
        $graphics.Dispose()
    }
}

function New-PanelBitmap {
    param(
        [Parameter(Mandatory = $true)]
        [System.Drawing.Image]$Source,
        [Parameter(Mandatory = $true)]
        [int]$Width,
        [Parameter(Mandatory = $true)]
        [int]$Height,
        [Parameter(Mandatory = $true)]
        [double]$ScaleFactor,
        [Parameter(Mandatory = $true)]
        [System.Drawing.Color]$BackgroundColor
    )

    $bitmap = New-Object System.Drawing.Bitmap($Width, $Height, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
        $graphics.Clear($BackgroundColor)

        $usableWidth = [Math]::Max(1.0, $Width * $ScaleFactor)
        $usableHeight = [Math]::Max(1.0, $Height * $ScaleFactor)
        $scale = [Math]::Min($usableWidth / [double]$Source.Width, $usableHeight / [double]$Source.Height)
        $drawWidth = [int][Math]::Round($Source.Width * $scale)
        $drawHeight = [int][Math]::Round($Source.Height * $scale)
        $offsetX = [int][Math]::Floor(($Width - $drawWidth) / 2.0)
        $offsetY = [int][Math]::Floor(($Height - $drawHeight) / 2.0)
        $targetRect = New-Object System.Drawing.Rectangle($offsetX, $offsetY, $drawWidth, $drawHeight)
        $graphics.DrawImage($Source, $targetRect)

        return $bitmap
    }
    finally {
        $graphics.Dispose()
    }
}

function Write-IcoFile {
    param(
        [Parameter(Mandatory = $true)]
        [System.Drawing.Image]$Source,
        [Parameter(Mandatory = $true)]
        [string]$OutputPath,
        [int[]]$Sizes = @(16, 32, 48, 64, 128, 256)
    )

    $frames = New-Object System.Collections.Generic.List[object]
    foreach ($size in $Sizes) {
        $bitmap = New-SquareBitmap -Source $Source -Size $size
        $memory = New-Object System.IO.MemoryStream
        try {
            $bitmap.Save($memory, [System.Drawing.Imaging.ImageFormat]::Png)
            $frames.Add([pscustomobject]@{
                Size = $size
                Data = $memory.ToArray()
            })
        }
        finally {
            $memory.Dispose()
            $bitmap.Dispose()
        }
    }

    $stream = [System.IO.File]::Open($OutputPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    $writer = New-Object System.IO.BinaryWriter($stream)
    try {
        $writer.Write([UInt16]0)
        $writer.Write([UInt16]1)
        $writer.Write([UInt16]$frames.Count)

        $offset = 6 + ($frames.Count * 16)
        foreach ($frame in $frames) {
            $dimension = if ($frame.Size -ge 256) { 0 } else { [byte]$frame.Size }
            $writer.Write([byte]$dimension)
            $writer.Write([byte]$dimension)
            $writer.Write([byte]0)
            $writer.Write([byte]0)
            $writer.Write([UInt16]1)
            $writer.Write([UInt16]32)
            $writer.Write([UInt32]$frame.Data.Length)
            $writer.Write([UInt32]$offset)
            $offset += $frame.Data.Length
        }

        foreach ($frame in $frames) {
            $writer.Write([byte[]]$frame.Data)
        }
    }
    finally {
        $writer.Dispose()
        $stream.Dispose()
    }
}

$resolvedSource = (Resolve-Path -Path $SourcePng -ErrorAction Stop).Path
$resolvedOutputDir = [System.IO.Path]::GetFullPath($OutputDir)
[System.IO.Directory]::CreateDirectory($resolvedOutputDir) | Out-Null

$runtimePngPath = Join-Path $resolvedOutputDir "$BaseName.png"
$iconPath = Join-Path $resolvedOutputDir "$BaseName.ico"
$headerBmpPath = Join-Path $resolvedOutputDir "$BaseName-header.bmp"
$wizardBmpPath = Join-Path $resolvedOutputDir "$BaseName-wizard.bmp"

$sourceImage = [System.Drawing.Image]::FromFile($resolvedSource)
try {
    Copy-Item -LiteralPath $resolvedSource -Destination $runtimePngPath -Force

    Write-IcoFile -Source $sourceImage -OutputPath $iconPath

    $headerBitmap = New-PanelBitmap -Source $sourceImage -Width 150 -Height 57 -ScaleFactor 0.78 -BackgroundColor ([System.Drawing.Color]::White)
    try {
        $headerBitmap.Save($headerBmpPath, [System.Drawing.Imaging.ImageFormat]::Bmp)
    }
    finally {
        $headerBitmap.Dispose()
    }

    $wizardBitmap = New-PanelBitmap -Source $sourceImage -Width 164 -Height 314 -ScaleFactor 0.82 -BackgroundColor ([System.Drawing.Color]::White)
    try {
        $wizardBitmap.Save($wizardBmpPath, [System.Drawing.Imaging.ImageFormat]::Bmp)
    }
    finally {
        $wizardBitmap.Dispose()
    }
}
finally {
    $sourceImage.Dispose()
}

Write-Host ("Generated branding assets in {0}" -f $resolvedOutputDir)
