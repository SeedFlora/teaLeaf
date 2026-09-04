# Re-install the test APK and run the latency benchmark, capturing thermal context.
#
# Thermal state is recorded before and after rather than assumed. The device charges over
# the same USB cable that carries ADB, so it warms during a session; every variant is
# measured under the same conditions and the absolute figures are reported as
# "under moderate thermal load" rather than as a cool-device best case.
param([string]$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path)

$ErrorActionPreference = "Continue"
$pkg = "org.tealeafai.screening"

if (-not (Get-Command java -ErrorAction SilentlyContinue)) {
    throw "Java was not found. Set JAVA_HOME and add its bin directory to PATH."
}

function Get-ThermalContext {
    $status = (adb shell dumpsys thermalservice 2>&1 |
        Select-String "Thermal Status" | ForEach-Object { $_.Line.Trim() }) -join "; "
    $batt = adb shell dumpsys battery 2>&1
    $temp = ($batt | Select-String "^\s+temperature:" | Select-Object -First 1) -replace '\D', ''
    $level = ($batt | Select-String "^\s+level:" | Select-Object -First 1) -replace '\D', ''
    [pscustomobject]@{
        thermal_status  = $status
        battery_temp_c  = if ($temp) { [double]$temp / 10 } else { $null }
        battery_level   = $level
        captured        = (Get-Date).ToUniversalTime().ToString("o")
    }
}

Push-Location "$Repo\android"
& .\gradlew.bat :app:installDebugAndroidTest --no-daemon 2>&1 |
    Select-String -Pattern "BUILD|FAILURE|^e: file" | ForEach-Object { $_.Line }
Pop-Location

$before = Get-ThermalContext
Write-Host "`nthermal BEFORE: $($before.thermal_status)  battery $($before.battery_temp_c) C  level $($before.battery_level)%"

Write-Host "`nrunning benchmark..."
adb shell am instrument -w -r `
    -e class org.tealeafai.screening.bench.LatencyBenchmarkTest `
    "$pkg.test/androidx.test.runner.AndroidJUnitRunner" 2>&1 |
    Select-String -Pattern "BENCH |OK \(|FAILURES|Tests run|stack=" | ForEach-Object { $_.Line.Trim() }

$after = Get-ThermalContext
Write-Host "`nthermal AFTER : $($after.thermal_status)  battery $($after.battery_temp_c) C  level $($after.battery_level)%"

$dest = "$Repo\reports\device_benchmark"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
adb pull "/sdcard/Android/data/$pkg/files/benchmark_report.json" "$dest\benchmark_report.json" 2>&1 |
    Select-String "pulled" | ForEach-Object { $_.Line.Trim() }

@{ before = $before; after = $after } | ConvertTo-Json -Depth 4 |
    Out-File "$dest\thermal_context.json" -Encoding utf8

Write-Host "`nreport: $((Get-Item "$dest\benchmark_report.json" -ErrorAction SilentlyContinue).Length) bytes"
