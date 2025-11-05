# Agent 1 — Implementation Plan (Step‑by‑Step)

> Branch: `agent1/inventory-restructure`  
> Write scope: `ops/output/**`, `ops/logs/**` only.

## 0) Bootstrap
```bash
git clone https://github.com/MathewYoussef/IMAS-portfolio
cd IMAS-portfolio
git checkout -b agent1/inventory-restructure
mkdir -p ops/output/inventory ops/output/proposals ops/logs
```

## 1) Repository tree & basic stats
```bash
# Full tree without .git
( command -v tree >/dev/null && tree -a -I ".git" > ops/output/inventory/tree.txt ) ||   find . -path ./.git -prune -o -print > ops/output/inventory/tree.txt

# File inventory (size, hash, type guess)
python3 - <<'PY' > ops/output/inventory/inventory.csv
import os, hashlib, mimetypes, subprocess, csv
from pathlib import Path
root=Path('.')
rows=[["path","type","size_bytes","sha1","last_commit","language","category","block_guess","action_suggestion","new_location","notes"]]
def sha1(p):
    h=hashlib.sha1()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()
def last_commit(p):
    try:
        return subprocess.check_output(["git","log","-1","--pretty=%h|%ad","--",p], text=True).strip()
    except: return ""
def guess_category(p):
    s=p.lower()
    if any(s.endswith(x) for x in [".ipynb",".rmd",".qmd"]): return "notebooks"
    if any(s.endswith(x) for x in [".py",".r",".m",".jl",".js",".ts",".cpp",".c",".h",".java"]): return "code"
    if any(x in s for x in ["/raw/","/rawdata/","/raw_data/"]): return "data/raw"
    if any(x in s for x in ["/processed/","/proc/"]): return "data/processed"
    if any(s.endswith(x) for x in [".csv",".tsv",".parquet",".feather",".xls",".xlsx"]): return "data"
    if any(s.endswith(x) for x in [".png",".jpg",".jpeg",".tif",".tiff",".gif",".mp4",".mov",".avi",".webm"]): return "media"
    if any(s.endswith(x) for x in [".md",".pdf",".tex"]): return "docs"
    if any(s.endswith(x) for x in [".yml",".yaml",".toml",".ini",".json",".cfg",".env",".lock"]): return "config"
    if any(x in s for x in ["/venv","/env","/conda","Pipfile","poetry.lock","requirements.txt"]): return "env"
    return "misc"
def block_guess(p):
    s=p.lower()
    if "reflectance" in s: return "Reflectance"
    if "initial_calibration" in s or "calibration" in s: return "Initial_Calibration"
    if "mamba" in s: return "Act_of_God_Mamba_Results"
    if "supplement" in s: return "Supplements"
    return ""
for dirpath,_,files in os.walk(root):
    if ".git" in dirpath: continue
    for f in files:
        p=os.path.join(dirpath,f).lstrip("./")
        try:
            st=os.stat(p); size=st.st_size; digest=sha1(p)
        except Exception as e:
            size=""; digest=""
        rows.append([p, mimetypes.guess_type(p)[0] or "", size, digest, last_commit(p), "", "", "", "", "", ""])
with open("ops/output/inventory/inventory.csv","w",newline="") as fh:
    csv.writer(fh).writerows(rows)
PY
```

## 2) Duplicates & large files
```bash
python3 - <<'PY'
import csv, collections
rows=list(csv.DictReader(open("ops/output/inventory/inventory.csv")))
byhash=collections.defaultdict(list)
for r in rows[1:]:
    byhash[r["sha1"]].append(r["path"])
dups=[["sha1","paths"]]+[[k,"; ".join(v)] for k,v in byhash.items() if k and len(v)>1]
open("ops/output/inventory/duplicates.csv","w").write("\n".join([",".join(x) for x in dups]))
large=[r for r in rows if r.get("size_bytes") and r["size_bytes"].isdigit() and int(r["size_bytes"])>90*1024*1024]
open("ops/output/inventory/large_files.csv","w").write("\n".join([r["path"] for r in large]))
PY
```

## 3) Mapping & proposal
- Create `ops/output/proposals/mapping.csv` with columns:  
  `old_path,new_path,action,rationale,risk`
- Create `ops/output/proposals/restructure_proposal.md` with:
  - Target top‑level: `hub/`, `Reflectance/`, `Initial_Calibration/`, `Act_of_God_Mamba_Results/`, `Supplements/`, `subprojects/`, `archive/`
  - For unrelated projects: propose extraction to `subprojects/<name>/` (or separate repos).

## 4) Issues & PR
- List issues to open in `ops/output/proposals/issues.md` (one line per path/problem).
- Open PR (reports‑only). Tag: `A1`, reviewers: project owner + Agents 2–4.
