"""Writes environment.json / environment.md: the PC description Section 4.1 needs (CPU, cores, RAM, OS, versions, data source)."""
import argparse, json, os, platform, subprocess, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="."); ap.add_argument("--jobs", default=""); ap.add_argument("--seeds", default="")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

def cpu_name():
    try:
        if sys.platform.startswith("win"):
            r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True, timeout=20)
            if r.stdout.strip(): return r.stdout.strip().splitlines()[0]
        elif sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        else:
            for l in open("/proc/cpuinfo"):
                if l.lower().startswith("model name"): return l.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"

def ram_gb():
    try:
        import psutil; return round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    try:
        if sys.platform.startswith("win"):
            r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"], capture_output=True, text=True, timeout=20)
            return round(int(r.stdout.strip()) / 2**30, 1)
        if sys.platform == "darwin":
            return round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30, 1)
        for l in open("/proc/meminfo"):
            if l.startswith("MemTotal"): return round(int(l.split()[1]) / 2**20, 1)
    except Exception:
        pass
    return None

def ver(m):
    try:
        import importlib; return importlib.import_module(m).__version__
    except Exception:
        return None

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
try:
    import common; src = common.DATA_SOURCE
except Exception as e:
    src = f"unknown ({e})"
info = dict(cpu=cpu_name(), logical_cores=os.cpu_count(), ram_gb=ram_gb(), os=platform.platform(), python=platform.python_version(),
            packages={m: ver(m) for m in ("numpy", "scipy", "sklearn", "xgboost", "pandas", "matplotlib", "pennylane")},
            data_source=src, jobs=a.jobs, seeds=a.seeds)
json.dump(info, open(out / "environment.json", "w"), indent=1)
p = info["packages"]
open(out / "environment.md", "w", encoding="utf-8").write(
    f"All experiments ran on a {info['cpu']} ({info['logical_cores']} logical cores, {info['ram_gb']} GB of RAM) under {info['os']}, "
    f"with Python {info['python']}, NumPy {p['numpy']}, SciPy {p['scipy']}, scikit-learn {p['sklearn']}, and XGBoost {p['xgboost']}; "
    f"the data came from: {info['data_source']}.\n")
print(open(out / "environment.md", encoding="utf-8").read())
