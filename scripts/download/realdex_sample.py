"""Download a few RealDex object archives from the authors' Google Drive links (gdown)."""
import json, sys, gdown, os
OUT = "raw_data/realdex/zips"
names = sys.argv[1:] or ["camera_param.json", "md5.txt", "cylinder.zip"]
links = {d["filename"]: d["url"] for d in json.load(open("external_code/RealDex/download/gdown_path.json"))}
os.makedirs(OUT, exist_ok=True)
for n in names:
    if not os.path.exists(f"{OUT}/{n}"):
        gdown.download(links[n], f"{OUT}/{n}", quiet=False)
print("DONE")
