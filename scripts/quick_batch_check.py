import sys, os, glob, requests
from collections import Counter

API = os.environ.get("AEGIS_API", "http://127.0.0.1:8000/detect")
paths = sys.argv[1:] or sorted([*glob.glob("*.jpg"), *glob.glob("*.png")])

def detect(p):
    with open(p, "rb") as f:
        r = requests.post(API, files={"file": f})
    r.raise_for_status()
    return r.json()

rows = []
for p in paths:
    j = detect(p)
    rows.append((p, j.get("confidence"), float(j.get("presence",0.0)),
                 float(j.get("presence_null",0.0)), float(j.get("margin",0.0))))

w = max([4]+[len(p) for p, *_ in rows])
print(f"{'file'.ljust(w)}  conf    presence    null        margin")
for p, c, pres, null, marg in rows:
    print(f"{p.ljust(w)}  {c:6}  {pres:10.6f}  {null:10.6f}  {marg:10.6f}")

ctr = Counter(c for _, c, *_ in rows)
print(f"\nSummary: {dict(ctr)}  total={len(rows)}")
