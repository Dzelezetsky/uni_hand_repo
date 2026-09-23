"""Seekable read-only HTTP file with Range requests, retries and a block LRU cache (for h5py on remote files)."""
import io, time
from collections import OrderedDict
import requests


class RemoteFile(io.RawIOBase):
    def __init__(self, url, block=1 << 20, max_blocks=512, offset=0):
        self.url, self.block, self.max_blocks, self.offset = url, block, max_blocks, offset
        self.s = requests.Session()
        r = self.s.head(url, allow_redirects=True, timeout=60)
        self.size = int(r.headers["Content-Length"]) - offset
        self.pos = 0
        self.cache = OrderedDict()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, pos, whence=0):
        self.pos = pos if whence == 0 else self.pos + pos if whence == 1 else self.size + pos
        return self.pos

    def _get(self, i):
        if i in self.cache:
            self.cache.move_to_end(i); return self.cache[i]
        a = self.offset + i * self.block
        b = min(a + self.block, self.offset + self.size) - 1
        for k in range(8):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120, allow_redirects=True)
                if r.status_code == 206 and len(r.content) == b - a + 1:
                    break
            except requests.RequestException:
                pass
            time.sleep(2 ** k)
        else:
            raise IOError(f"range fetch failed {a}-{b}")
        self.cache[i] = r.content
        if len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)
        return r.content

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        out, p = 0, self.pos
        while out < n:
            i, o = divmod(p, self.block)
            d = self._get(i)[o:o + n - out]
            buf[out:out + len(d)] = d
            out += len(d); p += len(d)
        self.pos = p
        return n
