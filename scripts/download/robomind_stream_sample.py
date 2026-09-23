"""Stream a RoboMIND task archive (<task>.tar.gz.part-aa, -ab, ... on HF, gated: needs `hf auth login`) and extract
only the first K episodes (each = <episode>/data/trajectory.hdf5). The gzip stream is read sequentially across parts
and abandoned as soon as K episodes are complete, so only a small prefix of the archive is downloaded.

usage: robomind_stream_sample.py <benchmark_dir> <embodiment_dir> <task> [K]
"""
import io
import os
import re
import sys
import tarfile

import requests
from huggingface_hub import HfApi, get_token, hf_hub_url

R = "x-humanoid-robomind/RoboMIND"
bench, emb, task = sys.argv[1:4]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 4
OUT = f"raw_data/robomind/{emb}/{task}"

parts = sorted(s.path for s in HfApi().list_repo_tree(R, path_in_repo=f"{bench}/{emb}", repo_type="dataset")
               if s.path.split("/")[-1].startswith(task + ".tar.gz.part-"))
headers = {"Authorization": f"Bearer {get_token()}"}


class PartsStream(io.RawIOBase):
    """Concatenation of the split parts as one sequential stream."""

    def __init__(self, paths):
        self.paths, self.resp, self.it = list(paths), None, None
        self.read_bytes = 0

    def readable(self):
        return True

    def _next(self):
        if not self.paths:
            return False
        url = hf_hub_url(R, self.paths.pop(0), repo_type="dataset")
        self.resp = requests.get(url, headers=headers, stream=True, timeout=120)
        self.resp.raise_for_status()
        self.it = self.resp.iter_content(1 << 20)
        self.buf = b""
        return True

    def readinto(self, b):
        while True:
            if self.it is None and not self._next():
                return 0
            if not self.buf:
                self.buf = next(self.it, b"")
                if not self.buf:
                    self.it = None
                    continue
            n = min(len(b), len(self.buf))
            b[:n] = self.buf[:n]
            self.buf = self.buf[n:]
            self.read_bytes += n
            return n


os.makedirs(OUT, exist_ok=True)
stream = PartsStream(parts)
done, current = [], None
with tarfile.open(fileobj=io.BufferedReader(stream, 1 << 22), mode="r|gz") as tf:
    for m in tf:
        mt = re.search(r"success_episodes/(train|val)/([^/]+)/", m.name)
        if mt and m.name.endswith("trajectory.hdf5") and m.isfile():
            tf.extract(m, OUT, filter="data")
            done.append(m.name)
            print(f"episode {len(done)}/{K}: {m.name} ({m.size / 1e6:.0f} MB), streamed {stream.read_bytes / 1e9:.2f} GB",
                  flush=True)
            if len(done) >= K:
                break
        elif m.isfile() and not m.name.endswith((".mp4", ".png", ".jpg")) and m.size < 5_000_000:
            tf.extract(m, OUT, filter="data")  # small side files (meta / json), if any
print("DONE", task, len(done), f"streamed {stream.read_bytes / 1e9:.2f} GB")
