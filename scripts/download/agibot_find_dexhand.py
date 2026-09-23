"""Find AgiBot World tasks recorded with the dexterous hand by probing the huge proprio_stats tars remotely.

For evenly spaced offsets in every proprio_stats/*.tar we locate the next proprio_stats.h5 member, read it
(~1 MB) and record (task, episode, effector/position shape). Output: raw_data/agibot/effector_probe.csv
"""
import csv
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import h5py
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(__file__))
from remote_tar import RemoteTar  # noqa: E402

STEP = int(float(sys.argv[1]) * 1e6) if len(sys.argv) > 1 else 150_000_000
OUT = "raw_data/agibot/effector_probe.csv"


def probe(args):
    repo, path, off = args
    t = RemoteTar(repo, path)
    for _ in range(4):  # skip directory entries
        m, hoff = t.find_header(off, window=1 << 21)
        if m is None:
            return None
        name, size, typ, data, nxt = m
        if name.endswith("proprio_stats.h5"):
            with h5py.File(io.BytesIO(t.read(data, size)), "r") as h:
                shp = h["state/effector/position"].shape if "state/effector/position" in h else None
                ts = h["timestamp"][:] if "timestamp" in h else []
            task, ep = name.split("/")[-3:-1]
            return dict(repo=repo, tar=path, offset=hoff, task=task, episode=ep, n=len(ts),
                        effector_dim=shp[1] if shp and len(shp) > 1 else None, h5_offset=data, h5_size=size)
        off = nxt
    return None


jobs = []
api = HfApi()
for repo in ("agibot-world/AgiBotWorld-Alpha", "agibot-world/AgiBotWorld-Beta"):
    for s in api.list_repo_tree(repo, path_in_repo="proprio_stats", repo_type="dataset"):
        jobs += [(repo, s.path, o) for o in range(0, s.size - 10_000_000, STEP)]
print("probes:", len(jobs), flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f, ThreadPoolExecutor(16) as ex:
    w = None
    for i, r in enumerate(ex.map(lambda a: (lambda: probe(a))() if True else None, jobs)):
        if r is None:
            continue
        if w is None:
            w = csv.DictWriter(f, fieldnames=list(r)); w.writeheader()
        w.writerow(r); f.flush()
        if i % 100 == 0:
            print(i, r["task"], r["effector_dim"], flush=True)
print("DONE")
