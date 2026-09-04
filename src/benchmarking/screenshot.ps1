# Capture a device screenshot without corrupting the PNG.
#
# `adb exec-out screencap -p > file.png` looks correct but produces a broken file under
# PowerShell: the `>` operator writes through a text encoder, mangling binary bytes and
# translating LF to CRLF. Capturing on the device and pulling the file keeps the bytes
# untouched, which matters because these screenshots are evidence.
param(
    [Parameter(Mandatory = $true)][string]$OutPath,
    [string]$Label = ""
)

$remote = "/sdcard/Android/data/org.tealeafai.screening/files/_shot.png"
adb shell mkdir -p /sdcard/Android/data/org.tealeafai.screening/files 2>&1 | Out-Null
adb shell screencap -p $remote 2>&1 | Out-Null

$dir = Split-Path -Parent $OutPath
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

adb pull $remote $OutPath 2>&1 | Out-Null
adb shell rm -f $remote 2>&1 | Out-Null

if (Test-Path $OutPath) {
    $bytes = [System.IO.File]::ReadAllBytes($OutPath)
    $isPng = $bytes.Length -gt 8 -and $bytes[0] -eq 0x89 -and $bytes[1] -eq 0x50 -and
             $bytes[2] -eq 0x4E -and $bytes[3] -eq 0x47
    $w = [int]$bytes[16] * 16777216 + [int]$bytes[17] * 65536 + [int]$bytes[18] * 256 + [int]$bytes[19]
    $h = [int]$bytes[20] * 16777216 + [int]$bytes[21] * 65536 + [int]$bytes[22] * 256 + [int]$bytes[23]
    "{0}  {1:N0} bytes  valid_png={2}  {3}x{4}  {5}" -f (Split-Path -Leaf $OutPath), $bytes.Length, $isPng, $w, $h, $Label
} else {
    "FAILED to capture $OutPath"
}
