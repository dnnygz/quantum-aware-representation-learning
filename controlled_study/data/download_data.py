#!/usr/bin/env python3
"""Downloads Ionosphere, Sonar and Blood Transfusion from the UCI repository into data/uci/ and checks them against
the copies in data/mirror/. WDBC comes from scikit-learn and the synthetic set is generated, so they need no download.

  python data/download_data.py

The experiment code uses data/uci/ automatically when the three files exist there, and data/mirror/ otherwise.
If a check prints DIFFERENT, report it before running anything.
If the direct URLs fail, the second route (package `ucimlrepo`, dataset ids 52, 151, 176) is used as a fallback.
"""
import io, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "uci"; MIRROR = HERE / "mirror"; OUT.mkdir(exist_ok=True)
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
SPEC = {   # file: (url, uci id, has_header, label column handling)
    "ionosphere.csv": (BASE + "ionosphere/ionosphere.data", 52, False),
    "sonar.csv": (BASE + "undocumented/connectionist-bench/sonar/sonar.all-data", 151, False),
    "transfusion.csv": (BASE + "blood-transfusion/transfusion.data", 176, True),
}

def fetch_url(url, header):
    raw = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    return pd.read_csv(io.StringIO(raw), header=0 if header else None)

def fetch_ucimlrepo(uid):
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=uid)
    return pd.concat([d.data.features.reset_index(drop=True), d.data.targets.reset_index(drop=True)], axis=1)

def same(a, b):
    a, b = a.copy(), b.copy()
    if a.shape != b.shape: return False, f"shape {a.shape} vs {b.shape}"
    a.columns = b.columns = range(a.shape[1])
    num = list(range(a.shape[1] - 1))
    if not np.allclose(a[num].astype(float).values, b[num].astype(float).values, atol=1e-9): return False, "feature values differ"
    la, lb = a[a.shape[1] - 1].astype(str).values, b[b.shape[1] - 1].astype(str).values
    if not (la == lb).all():
        # labels may be coded differently (e.g. 'g'/'b' vs 1/0); accept if the partition is identical
        return (len(set(zip(la, lb))) == len(set(la)) == len(set(lb))), "label coding differs (partition compared)"
    return True, "identical"

ok_all = True
for fname, (url, uid, header) in SPEC.items():
    df = None
    for route in ("url", "ucimlrepo"):
        try:
            df = fetch_url(url, header) if route == "url" else fetch_ucimlrepo(uid)
            print(f"{fname}: downloaded via {route} -> {df.shape}"); break
        except Exception as e:
            print(f"{fname}: route {route} failed ({type(e).__name__}: {str(e)[:100]})")
    if df is None:
        ok_all = False; print(f"{fname}: NOT downloaded; the mirror copy will be used."); continue
    mir = pd.read_csv(MIRROR / fname, header=None)
    good, why = same(df, mir)
    print(f"{fname}: {'MATCHES the mirror copy' if good else 'DIFFERENT from the mirror copy'} ({why})")
    if good: mir.to_csv(OUT / fname, header=False, index=False)    # write in the exact format the code reads
    else:
        ok_all = False; df.to_csv(OUT / (fname + ".unchecked"), header=False, index=False)
print("\nRESULT:", "all three files verified and saved in data/uci/" if ok_all else "at least one file was not verified; data/uci/ is incomplete, so the code keeps using data/mirror/")
sys.exit(0 if ok_all else 1)
