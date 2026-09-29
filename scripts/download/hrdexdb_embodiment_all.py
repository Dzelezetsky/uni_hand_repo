"""Download ALL HRDexDB episodes of one embodiment (raw/, cam_param, C2R, grasp_result, object pose v2) with
N_CAMS videos (preferring cam 22641023, the one used for visual checks).
usage: hrdexdb_embodiment_all.py <embodiment> [n_cams]"""
import sys

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

R, OUT = "HRDexDB/HRDexDB", "raw_data/hrdexdb"
emb = sys.argv[1]
n_cams = int(sys.argv[2]) if len(sys.argv) > 2 else 1
for f in ["metadata/episodes.parquet", "metadata/cameras.parquet"]:
    hf_hub_download(R, f, repo_type="dataset", local_dir=OUT)
e = pd.read_parquet(f"{OUT}/metadata/episodes.parquet")
c = pd.read_parquet(f"{OUT}/metadata/cameras.parquet")
sub = e[(e.embodiment == emb) & e.has_raw]
pats = []
for _, ep in sub.iterrows():
    p = ep.episode_path
    pats += [f"{p}/raw/**", f"{p}/cam_param/*.json", f"{p}/C2R.npy", f"{p}/grasp_result.json"]
    if isinstance(ep.object_pose_path, str):
        pats.append(ep.object_pose_path)
    cams = sorted(c[c.episode_id == ep.episode_id].video_path)
    pats += ([v for v in cams if "22641023" in v] + [v for v in cams if "22641023" not in v])[:n_cams]
print(len(sub), "episodes,", sub.object_id.nunique(), "objects", flush=True)
snapshot_download(R, repo_type="dataset", allow_patterns=pats, local_dir=OUT, max_workers=8)
print("DONE")
