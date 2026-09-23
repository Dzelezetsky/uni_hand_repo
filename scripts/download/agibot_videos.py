"""Fetch the videos of specific AgiBot episodes from observations/<task>/<range>.tar by binary search on episode id
(members are ordered by episode; each episode starts with <ep>/videos/*.mp4, then thousands of depth PNGs)."""
import os
import sys

from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(__file__))
from remote_tar import RemoteTar  # noqa: E402

repo, task = sys.argv[1], sys.argv[2]
episodes = [int(e) for e in sys.argv[3:]]
tars = [s.path for s in HfApi().list_repo_tree(repo, path_in_repo=f"observations/{task}", repo_type="dataset")]


def ep_at(t, off):
    m, hoff = t.find_header(off, window=1 << 21)
    return (int(m[0].split("/")[0]), hoff) if m else (10 ** 12, t.size)


for ep in episodes:
    cands = [p for p in tars if int(p.split("/")[-1].split("-")[0]) <= ep <= int(p.split("/")[-1].split("-")[1][:-4])]
    for p in cands:
        t = RemoteTar(repo, p)
        lo, hi = 0, t.size
        while hi - lo > 4 << 20:  # smallest offset whose next header has episode >= ep
            mid = (lo + hi) // 2
            e, _ = ep_at(t, mid)
            if e < ep:
                lo = mid
            else:
                hi = mid
        _, off = ep_at(t, lo)
        found = False
        for _ in range(4000):
            name, size, typ, data, nxt = t.header_at(off)
            e = int(name.split("/")[0])
            if e > ep or (e == ep and "/depth/" in name):
                break
            if e == ep and name.endswith(".mp4"):
                dst = f"raw_data/agibot/observations/{task}/{name}"
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as f:
                    for o in range(0, size, 32 << 20):
                        f.write(t.read(data + o, min(32 << 20, size - o)))
                found = True
                print(ep, name, f"{size / 1e6:.1f}MB", flush=True)
            off = nxt
        if found:
            break
    else:
        print(ep, "NOT FOUND", flush=True)
print("DONE")
