"""Random access into a huge UNCOMPRESSED tar on the HF Hub via HTTP Range requests (gated repos: uses the saved token).

- header_at(off): parse the 512-byte tar header at `off` -> (name, size, data_offset, next_header_offset)
- find_header(off): first valid header at or after `off` (scans a window, checks ustar magic + checksum)
- read(off, n): raw bytes
"""
from __future__ import annotations

import time

import requests
from huggingface_hub import get_token, hf_hub_url


class RemoteTar:
    def __init__(self, repo, path, repo_type="dataset"):
        self.url = hf_hub_url(repo, path, repo_type=repo_type)
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {get_token()}"
        r = self.s.head(self.url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self.pax_name = None

    def read(self, off, n):
        for k in range(6):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={off}-{off + n - 1}"}, timeout=120)
                if r.status_code == 206:
                    return r.content
            except requests.RequestException:
                pass
            time.sleep(2 ** k)
        raise IOError(f"range read failed at {off}")

    @staticmethod
    def _valid(h: bytes) -> bool:
        if len(h) < 512 or h[257:262] != b"ustar":
            return False
        try:
            chk = int(h[148:156].split(b"\0")[0].strip() or b"0", 8)
        except ValueError:
            return False
        return chk == sum(h[:148]) + 8 * 32 + sum(h[156:512])

    @staticmethod
    def _parse(h: bytes, off: int):
        name = h[0:100].split(b"\0")[0].decode()
        prefix = h[345:500].split(b"\0")[0].decode()
        if prefix:
            name = prefix + "/" + name
        size = int(h[124:136].split(b"\0")[0].strip() or b"0", 8)
        typ = h[156:157]
        data = off + 512
        nxt = data + ((size + 511) // 512) * 512
        return name, size, typ, data, nxt

    def header_at(self, off):
        h = self.read(off, 512)
        if not self._valid(h):
            raise ValueError(f"no tar header at {off}")
        name, size, typ, data, nxt = self._parse(h, off)
        if typ in (b"x", b"g"):  # pax header: real name in the payload, member follows
            pax = self.read(data, size).decode(errors="replace")
            for line in pax.splitlines():
                if " path=" in line:
                    real = line.split(" path=", 1)[1]
                    m = self.header_at(nxt)
                    return (real,) + m[1:]
            return self.header_at(nxt)
        if typ == b"L":  # GNU long name
            real = self.read(data, size).split(b"\0")[0].decode()
            m = self.header_at(nxt)
            return (real,) + m[1:]
        return name, size, typ, data, nxt

    def find_header(self, off, window=1 << 22):
        off -= off % 512
        while off < self.size:
            buf = self.read(off, window + 512)
            for i in range(0, len(buf) - 511, 512):
                if self._valid(buf[i:i + 512]):
                    return self.header_at(off + i), off + i
            off += window
        return None, None
