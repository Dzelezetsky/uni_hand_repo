"""Download N Humanoid Everyday H1 episodes (LeRobot v2.1) spread across distinct tasks."""
import json, sys
from huggingface_hub import hf_hub_download

R = "USC-PSI-Lab/Humanoid-Everyday-H1"
OUT = "raw_data/humanoid_everyday_h1"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
for f in ["meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl"]:
    hf_hub_download(R, f, repo_type="dataset", local_dir=OUT)
eps = [json.loads(l) for l in open(f"{OUT}/meta/episodes.jsonl")]
seen, picked = set(), []
step = max(1, len(eps) // (N * 3))
for ep in eps[::step]:
    t = ep["tasks"][0]
    if t not in seen:
        seen.add(t); picked.append(ep["episode_index"])
    if len(picked) == N:
        break
for i in picked:
    ch = i // 1000
    hf_hub_download(R, f"data/chunk-{ch:03d}/episode_{i:06d}.parquet", repo_type="dataset", local_dir=OUT)
    hf_hub_download(R, f"videos/chunk-{ch:03d}/egocentric/episode_{i:06d}.mp4", repo_type="dataset", local_dir=OUT)
    print(i, flush=True)
print("DONE")
