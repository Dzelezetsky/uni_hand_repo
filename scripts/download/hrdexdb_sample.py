"""Download a sample of HRDexDB Inspire episodes (raw state, cam params, 4 cameras each) + robot URDF assets."""
import random, sys
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

R = "HRDexDB/HRDexDB"
OUT = "raw_data/hrdexdb"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_CAMS = 4

for f in ["metadata/episodes.parquet", "metadata/cameras.parquet", "metadata/embodiments.parquet"]:
    hf_hub_download(R, f, repo_type="dataset", local_dir=OUT)
e = pd.read_parquet(f"{OUT}/metadata/episodes.parquet")
c = pd.read_parquet(f"{OUT}/metadata/cameras.parquet")
pats = ["assets/robots/inspire/**", "assets/robots/inspire_f1/**", "assets/robots/xarm/**", "assets/robots/*.urdf"]
random.seed(0)
for emb in ["inspire_dftp", "inspire_f1"]:
    sub = e[(e.embodiment == emb) & e.has_raw]
    for o in random.sample(sorted(sub.object_id.unique()), N):
        ep = sub[sub.object_id == o].iloc[0]
        p = ep.episode_path
        pats += [f"{p}/raw/**", f"{p}/cam_param/*.json", f"{p}/C2R.npy", f"{p}/grasp_result.json"]
        if isinstance(ep.object_pose_path, str):
            pats.append(ep.object_pose_path)
        pats += sorted(c[c.episode_id == ep.episode_id].video_path)[:N_CAMS]
        print(emb, ep.episode_id, flush=True)
snapshot_download(R, repo_type="dataset", allow_patterns=pats, local_dir=OUT, max_workers=8)
print("DONE")
