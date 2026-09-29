"""RealDex poses-only scaling (user decision 2026-09-29: no images for the remaining objects, disk).
For each object: gdown the zip -> extract everything except images / point clouds -> delete the zip.
Objects whose extracted folder already exists are skipped; the 4 full zips already on disk are kept.
Google Drive quota failures are logged and skipped: just re-run later.
usage: realdex_poses_only.py [obj ...]   (default: all objects in gdown_path.json)"""
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import gdown

Z, OUT = Path("raw_data/realdex/zips"), Path("raw_data/realdex/extracted")
BAGS = OUT / "storage/group/4dvlab/youzhuo/bags"
MIN_FREE = 60 * 2**30  # bytes; largest RealDex zip is ~18 GB
links = {d["filename"]: d["url"] for d in json.load(open("external_code/RealDex/download/gdown_path.json"))}
objs = sys.argv[1:] or sorted(n[:-4] for n in links if n.endswith(".zip"))
for o in objs:
    if shutil.disk_usage(Z).free < MIN_FREE:
        print("STOP: free disk below", MIN_FREE // 2**30, "GiB", flush=True); break
    if (BAGS / o).exists():
        print(o, "already extracted", flush=True); continue
    zp = Z / f"{o}.zip"
    try:
        gdown.download(links[f"{o}.zip"], str(zp), quiet=True)
        z = zipfile.ZipFile(zp)
    except Exception as e:  # quota / network / truncated file
        print(o, "FAILED", repr(e)[:200], flush=True)
        zp.unlink(missing_ok=True)
        for part in Z.glob(f"{o}.zip*.part"):  # gdown temp file of the failed download
            part.unlink()
        continue
    k = 0
    for i in z.infolist():
        if i.is_dir() or i.filename.endswith((".png", ".jpg", ".pcd", ".ply")):
            continue
        z.extract(i, OUT); k += 1
    seqs = sorted({i.filename.split("/")[6] for i in z.infolist() if len(i.filename.split("/")) > 7})
    z.close()
    os.remove(zp)
    print(o, "extracted", k, "files, sequences:", len(seqs), "-> zip deleted", flush=True)
print("DONE")
