"""Download ALL HRDexDB episodes of given embodiment + objects: raw/, cam_param, C2R, grasp_result, object pose v2,
and N_CAMS videos. usage: hrdexdb_objects.py <embodiment> <n_cams> obj1 obj2 ..."""
import sys
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

R, OUT = "HRDexDB/HRDexDB", "raw_data/hrdexdb"
emb, n_cams, objs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
for f in ["metadata/episodes.parquet", "metadata/cameras.parquet"]:
    hf_hub_download(R, f, repo_type="dataset", local_dir=OUT)
e = pd.read_parquet(f"{OUT}/metadata/episodes.parquet")
c = pd.read_parquet(f"{OUT}/metadata/cameras.parquet")
sub = e[(e.embodiment == emb) & e.object_id.isin(objs) & e.has_raw]
pats = []
for _, ep in sub.iterrows():
    p = ep.episode_path
    pats += [f"{p}/raw/**", f"{p}/cam_param/*.json", f"{p}/C2R.npy", f"{p}/grasp_result.json"]
    if isinstance(ep.object_pose_path, str):
        pats.append(ep.object_pose_path)
    cams = sorted(c[c.episode_id == ep.episode_id].video_path)
    pats += [v for v in cams if "22641023" in v][:n_cams] or cams[:n_cams]
print(len(sub), "episodes:", sub.groupby("object_id").size().to_dict(), flush=True)
snapshot_download(R, repo_type="dataset", allow_patterns=pats, local_dir=OUT, max_workers=8)
print("DONE")
