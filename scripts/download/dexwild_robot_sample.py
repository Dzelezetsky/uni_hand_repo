"""Copy the first N robot episodes of a DexWild task from the remote tar'ed HDF5 (HDF5 superblock at tar offset 512)
into a local HDF5 via HTTP Range requests (no full download).

Low-dimensional groups are copied for all N episodes. Camera groups (one small dataset per JPEG frame) are copied
only for the first N_IMG episodes, fetching frame bytes with parallel range requests using each dataset's file offset.
"""
import os, sys
from concurrent.futures import ThreadPoolExecutor
import h5py, numpy as np, requests
from huggingface_hub import hf_hub_url
sys.path.insert(0, os.path.dirname(__file__))
from remote_file import RemoteFile

TASK = sys.argv[1] if len(sys.argv) > 1 else "pour"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
N_IMG = int(sys.argv[3]) if len(sys.argv) > 3 else 3
TAR_HEADER = 512
OUT = f"raw_data/dexwild/robot_{TASK}_first{N}.hdf5"
url = hf_hub_url("boardd/dexwild-dataset", f"{TASK}_data/robot/robot_{TASK}_data.part_00", repo_type="dataset")
src = h5py.File(RemoteFile(url, block=4 << 20, max_blocks=128), "r")
sess = requests.Session()


def fetch(off, n):
    a = TAR_HEADER + off
    for _ in range(6):
        try:
            r = sess.get(url, headers={"Range": f"bytes={a}-{a + n - 1}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == n:
                return r.content
        except requests.RequestException:
            pass
    raise IOError(f"failed {a}")


def copy_images(sg, dg):
    items = []
    for name in sg:
        d = sg[name]
        off = d.id.get_offset()
        if off is None or d.chunks is not None or d.compression:
            items.append((name, None, d))
        else:
            items.append((name, (off, d.id.get_storage_size()), d))
    with ThreadPoolExecutor(32) as ex:
        blobs = list(ex.map(lambda it: fetch(*it[1]) if it[1] else it[2][()], items))
    for (name, loc, d), b in zip(items, blobs):
        arr = np.frombuffer(b, dtype=d.dtype).reshape(d.shape) if loc else b
        dg.create_dataset(name, data=arr)


def copy_arrays(obj, parent, name):
    """Read-then-write copy (h5py's H5Ocopy issues tiny reads, which is very slow over HTTP)."""
    if isinstance(obj, h5py.Dataset):
        parent.create_dataset(name, data=obj[()])
    else:
        g = parent.create_group(name)
        for k in obj:
            copy_arrays(obj[k], g, k)
    for a, v in obj.attrs.items():
        parent[name].attrs[a] = v


os.makedirs(os.path.dirname(OUT), exist_ok=True)
with h5py.File(OUT, "a") as dst:
    dst.attrs["source_url"] = url
    dst.attrs["source_num_episodes"] = len(src.keys())
    eps = sorted(src.keys())[:N]
    jobs = [(i, k, sub, False) for i, k in enumerate(eps) for sub in src[k]]
    jobs = [j for j in jobs if not (j[2].endswith("_cam") or j[2] == "zed_obs")] + \
           [(i, k, sub, True) for i, k, sub, _ in jobs if (sub.endswith("_cam") or sub == "zed_obs") and i < N_IMG]
    for i, k, sub, is_cam in jobs:  # all low-dim streams first, then camera frames
        eg = dst.require_group(k)
        if sub in eg:
            continue
        if is_cam:
            copy_images(src[k][sub], eg.create_group(sub))
        else:
            copy_arrays(src[k][sub], eg, sub)
        dst.flush()
        print("copied", k, sub, flush=True)
src.close()
print("DONE")
