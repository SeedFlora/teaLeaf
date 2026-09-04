"""Collect a read-only hardware and software inventory of the connected Android device.

Every command issued here is a non-destructive read: property lookups, /proc
queries and dumpsys reads. Nothing is installed, written, deleted or configured,
and no personal storage is touched.

Hardware facts are read from the device itself rather than inferred from the
commercial model name, so that the manuscript reports what was measured.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ADB = "adb"


def sh(args: list[str], timeout: int = 60) -> str:
    try:
        r = subprocess.run([ADB, *args], capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"<error: {type(exc).__name__}: {exc}>"


def prop(name: str) -> str:
    return sh(["shell", "getprop", name])


def main() -> int:
    out_path = Path(sys.argv[1])

    devices = sh(["devices", "-l"])
    lines = [l for l in devices.splitlines()[1:] if l.strip()]
    if not lines:
        print("NO DEVICE CONNECTED", file=sys.stderr)
        return 2

    props = {
        "manufacturer": "ro.product.manufacturer",
        "brand": "ro.product.brand",
        "model": "ro.product.model",
        "device": "ro.product.device",
        "name": "ro.product.name",
        "android_release": "ro.build.version.release",
        "api_level": "ro.build.version.sdk",
        "security_patch": "ro.build.version.security_patch",
        "build_id": "ro.build.id",
        "build_fingerprint": "ro.build.fingerprint",
        "build_type": "ro.build.type",
        "soc_manufacturer": "ro.soc.manufacturer",
        "soc_model": "ro.soc.model",
        "board_platform": "ro.board.platform",
        "hardware": "ro.hardware",
        "cpu_abi": "ro.product.cpu.abi",
        "cpu_abilist": "ro.product.cpu.abilist",
        "opengles_version": "ro.opengles.version",
        "vulkan_level": "ro.hardware.vulkan.level",
        "dalvik_heapsize": "dalvik.vm.heapsize",
    }
    info = {k: prop(v) for k, v in props.items()}

    # --- memory ---
    meminfo = sh(["shell", "cat", "/proc/meminfo"])
    mem: dict[str, int] = {}
    for line in meminfo.splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s+kB", line)
        if m and m.group(1) in ("MemTotal", "MemAvailable", "MemFree", "SwapTotal"):
            mem[m.group(1)] = int(m.group(2))

    # --- cpu topology ---
    cpuinfo = sh(["shell", "cat", "/proc/cpuinfo"])
    n_cores = len(re.findall(r"^processor", cpuinfo, re.MULTILINE))
    max_freqs = sh(["shell",
                    "for f in /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq; do cat $f 2>/dev/null; done"])
    freqs_khz = [int(x) for x in max_freqs.split() if x.isdigit()]

    # --- display ---
    size = sh(["shell", "wm", "size"])
    density = sh(["shell", "wm", "density"])

    # --- thermal ---
    thermal_raw = sh(["shell",
                      "for z in /sys/class/thermal/thermal_zone*; do "
                      "echo \"$(cat $z/type 2>/dev/null)=$(cat $z/temp 2>/dev/null)\"; done"])
    thermal: dict[str, float] = {}
    for entry in thermal_raw.splitlines():
        if "=" in entry:
            t, v = entry.split("=", 1)
            v = v.strip()
            if v.lstrip("-").isdigit():
                val = int(v)
                # Kernels report either milli-degrees or degrees Celsius.
                thermal[t.strip()] = val / 1000.0 if abs(val) > 1000 else float(val)

    thermal_service = sh(["shell", "dumpsys", "thermalservice"])
    thermal_status = ""
    m = re.search(r"Thermal Status:\s*(\S+)", thermal_service)
    if m:
        thermal_status = m.group(1)

    # --- battery (read-only) ---
    battery = sh(["shell", "dumpsys", "battery"])
    bat: dict[str, str] = {}
    for line in battery.splitlines():
        if ":" in line:
            k, _, v = line.strip().partition(":")
            if k.strip() in ("level", "temperature", "health", "status", "AC powered",
                             "USB powered", "voltage", "Charge counter"):
                bat[k.strip()] = v.strip()

    # --- acceleration / runtime availability ---
    nnapi_features = sh(["shell", "pm", "list", "features"])
    accel = {
        "neuralnetworks_feature": "android.hardware.neuralnetworks" in nnapi_features,
        "vulkan_hardware_version": "android.hardware.vulkan.version" in nnapi_features,
        "vulkan_compute": "android.hardware.vulkan.compute" in nnapi_features,
        "opengles_version_prop": info.get("opengles_version", ""),
        "note": (
            "Presence of the android.hardware.neuralnetworks feature indicates an NNAPI "
            "implementation is declared. Whether a given TFLite delegate actually accelerates "
            "this specific model is determined empirically during the G7 benchmark, not "
            "inferred from this flag."
        ),
    }

    inventory = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "adb_devices_raw": devices,
        "device_identity": info,
        "cpu": {
            "cores_reported": n_cores,
            "abi": info.get("cpu_abi", ""),
            "abilist": info.get("cpu_abilist", ""),
            "max_freq_mhz_per_core": [round(f / 1000) for f in freqs_khz],
        },
        "memory_kb": mem,
        "memory_total_gb": round(mem.get("MemTotal", 0) / 1024 / 1024, 2) if mem else None,
        "memory_available_gb": round(mem.get("MemAvailable", 0) / 1024 / 1024, 2) if mem else None,
        "display": {"size": size, "density": density},
        "thermal_zones_c": thermal,
        "thermal_status": thermal_status,
        "battery": bat,
        "acceleration": accel,
        "collection_method": (
            "Read-only adb queries: getprop, /proc/meminfo, /proc/cpuinfo, sysfs cpufreq and "
            "thermal nodes, dumpsys battery/thermalservice, pm list features. No device state "
            "was modified."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    d = inventory["device_identity"]
    print(f"Device      : {d['manufacturer']} {d['model']} ({d['device']})")
    print(f"Android     : {d['android_release']} (API {d['api_level']}), patch {d['security_patch']}")
    print(f"SoC         : {d['soc_manufacturer']} {d['soc_model']}  platform={d['board_platform']}")
    print(f"CPU         : {n_cores} cores, ABI {d['cpu_abi']}")
    print(f"             max freqs (MHz): {inventory['cpu']['max_freq_mhz_per_core']}")
    print(f"RAM         : {inventory['memory_total_gb']} GB total, {inventory['memory_available_gb']} GB available")
    print(f"Display     : {size} | {density}")
    print(f"Thermal     : status={thermal_status or 'n/a'}, {len(thermal)} zones read")
    print(f"NNAPI feat  : {accel['neuralnetworks_feature']}")
    print(f"Battery     : level={bat.get('level','?')} temp={bat.get('temperature','?')} status={bat.get('status','?')}")
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
