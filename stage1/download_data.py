"""Download the Stage-1 datasets into raw_data/ with the layout the UnifiedDex adapters expect.

Hand / state data and metadata of EVERY episode are downloaded (small; needed for canonical geometry and labels);
RGB only for the primary camera, all video files by default (~505 GB video, plan for a ~1.1 TB server):
  humanoid_everyday_h1  egocentric      openarm_banana  head      trex            head_left
  dexora                top             egosteer        head      sharpa_origami  head_left
A fraction < 1 (--fraction ds=f) keeps every k-th video file (EgoSteer files are task-sorted -> all tasks stay) or
every k-th group (Origami: season). Depth, other cameras, wrist views are never downloaded. Already present files
are skipped (safe to re-run). Needs `hf auth login` with access to the gated repos (Origami, OpenArm Banana).
usage: python stage1/download_data.py [--dry-run] [--only ds1,ds2] [--fraction ds=f ...]
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

from huggingface_hub import HfApi, snapshot_download

SPEC = {
    "humanoid_everyday_h1": dict(repo="USC-PSI-Lab/Humanoid-Everyday-H1", out="raw_data/humanoid_everyday_h1",
                                 fraction=1.0, keep=[r"^meta/", r"^data/"], video=r"^videos/[^/]+/egocentric/"),
    "openarm_banana": dict(repo="June777/openarm_banana_all_1072episodes", out="raw_data/openarm_banana/all_1072",
                           fraction=1.0, keep=[r"^meta/", r"^data/", r"^README\.md$", r"^build_all_1072\.py$"],
                           video=r"^videos/[^/]+/observation\.images\.head/"),
    "trex": dict(repo="zekaiwang/trex_dataset", out="raw_data/trex", fraction=1.0,
                 keep=[r"^meta/", r"^data/", r"^README\.md$", r"^LICENSE", r"^episodes_preview\.parquet$"],
                 video=r"^videos/observation\.images\.head_left/"),
    "dexora": dict(repo="Dexora/Dexora_Real-World_Dataset", out="raw_data/dexora", fraction=1.0,
                   keep=[r"^airbot_[a-z_]+/meta/", r"^airbot_[a-z_]+/data/", r"^README\.md$"],
                   video=r"^airbot_[a-z_]+/videos/[^/]+/observation\.images\.top/"),
    "egosteer": dict(repo="EgoSteer/EgoSteer-RealWorld", out="raw_data/egosteer", fraction=1.0,
                     keep=[r"^meta/", r"^data/", r"^README\.md$"], video=r"^videos/observation\.images\.head/"),
    "sharpa_origami": dict(repo="SharpaIT/Robotic_Origami_Challenge", out="raw_data/origami", fraction=1.0,
                           keep=[r"^season_[^/]+/lerobot3\.0/meta/", r"^season_[^/]+/lerobot3\.0/data/"],
                           video=r"^season_[^/]+/lerobot3\.0/videos/observation\.images\.head_left/",
                           video_group=r"^(season_[^/]+)/"),
}


def pick(files, fraction, group_re=None):
    """Deterministic subset: every k-th file (or every k-th group, e.g. Origami season) in sorted order."""
    if fraction >= 1.0:
        return files
    k = max(1, round(1 / fraction))
    if group_re is None:
        return [f for i, f in enumerate(sorted(files)) if i % k == 0]
    groups = sorted({re.match(group_re, f).group(1) for f in files})
    keep = {g for i, g in enumerate(groups) if i % k == 0}
    return [f for f in files if re.match(group_re, f).group(1) in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--fraction", nargs="*", default=[], help="override, e.g. egosteer=0.5")
    a = ap.parse_args()
    over = {k: float(v) for k, v in (x.split("=") for x in a.fraction)}
    api = HfApi()
    grand = defaultdict(float)
    for ds, s in SPEC.items():
        if a.only and ds not in a.only.split(","):
            continue
        frac = over.get(ds, s["fraction"])
        tree = [(f.path, f.size) for f in api.list_repo_tree(s["repo"], repo_type="dataset", recursive=True)
                if getattr(f, "size", None) is not None]
        size = dict(tree)
        keep = [p for p, _ in tree if any(re.search(r, p) for r in s["keep"])]
        vids = [p for p, _ in tree if re.search(s["video"], p)]
        vids_sel = pick(vids, frac, s.get("video_group"))
        gb = lambda ps: sum(size[p] for p in ps) / 1e9
        print(f"{ds:22s} data+meta {len(keep):6d} files {gb(keep):7.1f} GB | video {len(vids_sel):6d}/{len(vids)} "
              f"files {gb(vids_sel):7.1f}/{gb(vids):7.1f} GB (fraction {frac})", flush=True)
        grand["data"] += gb(keep)
        grand["video"] += gb(vids_sel)
        if not a.dry_run:
            snapshot_download(s["repo"], repo_type="dataset", local_dir=s["out"], allow_patterns=keep + vids_sel,
                              max_workers=8)
            print(f"{ds}: done -> {s['out']}", flush=True)
    print(f"TOTAL data+meta {grand['data']:.1f} GB, video {grand['video']:.1f} GB")


if __name__ == "__main__":
    main()
