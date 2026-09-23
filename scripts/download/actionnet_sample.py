"""Sample Fourier ActionNet episodes straight out of the big uncompressed tars (HTTP range reads, gated repo).

For each listed tar: walk tar headers from the start and fetch the first K episodes' <id>.hdf5, <id>/*/rgb.mp4 and
<id>/*/timestamps.json (depth.mkv skipped). usage: actionnet_sample.py K tar1 [tar2 ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from remote_tar import RemoteTar  # noqa: E402

R = "FourierIntelligence/ActionNet"
OUT = "raw_data/actionnet"
K = int(sys.argv[1])
for tar in sys.argv[2:]:
    t = RemoteTar(R, tar)
    off, eps = 0, []
    while off < t.size:
        name, size, typ, data, nxt = t.header_at(off)
        off = nxt
        ep = name.split("/")[0].replace(".hdf5", "")
        if ep not in eps:
            if len(eps) == K:
                break
            eps.append(ep)
        if size and name.endswith((".hdf5", "rgb.mp4", "timestamps.json")):
            dst = os.path.join(OUT, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not (os.path.exists(dst) and os.path.getsize(dst) == size):
                with open(dst, "wb") as f:
                    for o in range(0, size, 32 << 20):
                        f.write(t.read(data + o, min(32 << 20, size - o)))
            print(tar, name, f"{size / 1e6:.1f}MB", flush=True)
print("DONE")
