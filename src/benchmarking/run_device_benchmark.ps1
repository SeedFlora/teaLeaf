# Run the on-device latency benchmark with the correct ordering.
#
# The obvious approach -- push models, then `gradlew connectedAndroidTest` -- silently
# fails. That Gradle task reinstalls the APK, which clears /sdcard/Android/data/<pkg>/,
# so the models are gone before the test opens the directory.
#
# Correct order: install both APKs, THEN push the models, THEN invoke the instrumentation
# directly with `am instrument` so nothing reinstalls in between, THEN pull the report
# before anything can clean up.
param(
    [string]$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [switch]$SkipInstall
)

# adb writes transfer progress to stderr, and under ErrorActionPreference=Stop PowerShell
# treats that as a terminating failure even when the push succeeded. Errors are therefore
# checked explicitly below rather than by preference.
$ErrorActionPreference = "Continue"
$pkg = "org.tealeafai.screening"
$remoteDir = "/sdcard/Android/data/$pkg/files/benchmark_models"
$runner = "$pkg.test/androidx.test.runner.AndroidJUnitRunner"

if (-not (Get-Command java -ErrorAction SilentlyContinue)) {
    throw "Java was not found. Set JAVA_HOME and add its bin directory to PATH."
}

if (-not $SkipInstall) {
    Write-Host "=== installing app and test APKs ==="
    Push-Location "$Repo\android"
    & .\gradlew.bat :app:installDebug :app:installDebugAndroidTest --no-daemon 2>&1 |
        Select-String -Pattern "BUILD|Installing|Installed|FAILURE" | ForEach-Object { $_.Line }
    Pop-Location
}

Write-Host "`n=== pushing model variants (after install, so they survive) ==="
adb shell mkdir -p $remoteDir 2>&1 | Out-Null
$pushed = 0
Get-ChildItem "$Repo\models\exported\*.tflite" | ForEach-Object {
    adb push $_.FullName "$remoteDir/$($_.Name)" 2>&1 | Out-Null
    $pushed++
    "  pushed {0}  {1:N2} MB" -f $_.Name, ($_.Length / 1MB)
}
Write-Host "  $pushed models on device"

$onDevice = (adb shell ls $remoteDir 2>&1 | Where-Object { $_ -match "\.tflite" }).Count
if ($onDevice -lt 1) { throw "no models present on device after push; aborting" }

Write-Host "`n=== thermal state before benchmark ==="
adb shell dumpsys thermalservice 2>&1 | Select-String "Thermal Status" | ForEach-Object { $_.Line.Trim() }

Write-Host "`n=== running benchmark instrumentation ==="
$out = adb shell am instrument -w -r `
    -e class org.tealeafai.screening.bench.LatencyBenchmarkTest `
    $runner 2>&1
$out | Select-String -Pattern "BENCH |BENCHMARK_REPORT_PATH|OK \(|FAILURES|Error|AssertionError" |
    ForEach-Object { $_.Line.Trim() }

Write-Host "`n=== pulling report before anything can clear app storage ==="
$dest = "$Repo\reports\device_benchmark"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
adb pull "/sdcard/Android/data/$pkg/files/benchmark_report.json" "$dest\benchmark_report.json" 2>&1 |
    Select-String "pulled|error" | ForEach-Object { $_.Line.Trim() }

if (Test-Path "$dest\benchmark_report.json") {
    $size = (Get-Item "$dest\benchmark_report.json").Length
    Write-Host "report: $size bytes"
    if ($size -lt 200) { throw "benchmark report is suspiciously small; it likely measured nothing" }
} else {
    throw "benchmark report was not produced"
}

Write-Host "`n=== thermal state after benchmark ==="
adb shell dumpsys thermalservice 2>&1 | Select-String "Thermal Status" | ForEach-Object { $_.Line.Trim() }
