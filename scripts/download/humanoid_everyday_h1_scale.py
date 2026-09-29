"""Scale Humanoid Everyday H1: up to PER_TASK episodes of EVERY task (egocentric video + parquet), task-balanced.
usage: humanoid_everyday_h1_scale.py [per_task]   (already downloaded files are skipped)"""
import json
import random
import sys

from huggingface_hub import hf_hub_download, snapshot_download

R, OUT = "USC-PSI-Lab/Humanoid-Everyday-H1", "raw_data/humanoid_everyday_h1"
PER_TASK = int(sys.argv[1]) if len(sys.argv) > 1 else 4
for f in ["meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl"]:
    hf_hub_download(R, f, repo_type="dataset", local_dir=OUT)
eps = [json.loads(l) for l in open(f"{OUT}/meta/episodes.jsonl")]
by_task = {}
for ep in eps:
    by_task.setdefault(ep["tasks"][0], []).append(ep["episode_index"])
random.seed(0)
picked = sorted(i for ids in by_task.values() for i in random.sample(ids, min(PER_TASK, len(ids))))
pats = []
for i in picked:
    ch = i // 1000
    pats += [f"data/chunk-{ch:03d}/episode_{i:06d}.parquet", f"videos/chunk-{ch:03d}/egocentric/episode_{i:06d}.mp4"]
print(f"{len(by_task)} tasks, {len(picked)} episodes", flush=True)
snapshot_download(R, repo_type="dataset", allow_patterns=pats, local_dir=OUT, max_workers=8)
print("DONE")
