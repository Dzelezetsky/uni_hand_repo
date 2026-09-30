"""EgoSteer-RealWorld sample: meta + the first data parquet (+ matching head/chest RGB files, no depth).
usage: egosteer_sample.py [n_files] [--no-video]"""
import sys

from huggingface_hub import snapshot_download

R, OUT = "EgoSteer/EgoSteer-RealWorld", "raw_data/egosteer"
N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1
pats = ["meta/*", "meta/episodes/*/*.parquet"]
for i in range(N):
    pats.append(f"data/chunk-000/file-{i:03d}.parquet")
    if "--no-video" not in sys.argv:
        pats += [f"videos/observation.images.{c}/chunk-000/file-{i:03d}.mp4" for c in ("head", "chest")]
snapshot_download(R, repo_type="dataset", allow_patterns=pats, local_dir=OUT, max_workers=8)
print("DONE")
