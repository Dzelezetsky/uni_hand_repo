"""Extract everything except images/point clouds from downloaded RealDex object zips (images stay in the zip and
are referenced as zip://). usage: realdex_extract.py [obj ...]  (default: all complete zips)"""
import sys
import zipfile
from pathlib import Path

Z, OUT = Path("raw_data/realdex/zips"), Path("raw_data/realdex/extracted")
names = sys.argv[1:] or [p.stem for p in Z.glob("*.zip")]
for n in names:
    p = Z / f"{n}.zip"
    try:
        z = zipfile.ZipFile(p)
    except (zipfile.BadZipFile, FileNotFoundError) as e:
        print(n, "SKIP", e); continue
    k = 0
    for i in z.infolist():
        if i.is_dir() or i.filename.endswith((".png", ".jpg", ".pcd", ".ply")):
            continue
        if not (OUT / i.filename).exists():
            z.extract(i, OUT); k += 1
    seqs = sorted({i.filename.split("/")[6] for i in z.infolist() if len(i.filename.split("/")) > 7})
    print(n, "extracted", k, "files; sequences:", seqs, flush=True)
