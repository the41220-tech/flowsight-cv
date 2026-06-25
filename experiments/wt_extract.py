"""Sequential local-header ZIP extractor for the CORRUPT + STREAMING EPFL WILDTRACK zip.

The EPFL `Wildtrack_dataset_full.zip` is (a) >4GB with wrapped central-directory
offsets (unzip/7z/python-zipfile fail) AND (b) a STREAMING zip: local headers carry
csize=0 with the real size in a trailing data-descriptor. So we scan for PK\\x03\\x04
from byte 0 and decompress each member with a streaming raw-inflate (zlib.decompressobj),
which finds its own end -> no need for csize. We keep only what the MVDet 4-view
experiment needs: calibrations + annotations_positions + first --nframes frames of each
wanted view (Image_subsets/C{n}). Every deflated member is decompressed (to learn its
byte length) but only wanted ones are written.

Usage:
  python experiments/wt_extract.py --zip /content/WT.zip --out /content/WTx \
      --views C1,C2,C4,C5 --nframes 400
"""
import argparse
import os
import struct
import zlib

SIG = b"PK\x03\x04"


def want(name, views, nframes, counts):
    if name.endswith("/"):
        return False
    if "calibrations/" in name or "annotations_positions/" in name:
        return True
    for v in views:
        if ("Image_subsets/%s/" % v) in name:
            return counts.get(v, 0) < nframes
    return False


def inflate_stream(f, dstart):
    """Raw-inflate a member starting at dstart; return (raw_bytes, consumed_compressed)."""
    do = zlib.decompressobj(-15)
    f.seek(dstart)
    out = []
    consumed = 0
    while True:
        chunk = f.read(1 << 20)
        if not chunk:
            break
        out.append(do.decompress(chunk))
        if do.eof:
            consumed += len(chunk) - len(do.unused_data)
            break
        consumed += len(chunk)
    return b"".join(out), consumed


def main(a):
    views = a.views.split(",")
    counts = {}
    fsize = os.path.getsize(a.zip)
    f = open(a.zip, "rb")
    pos = 0
    nwrote = 0
    while pos < fsize:
        f.seek(pos)
        head = f.read(30)
        if len(head) < 30:
            break
        if head[:4] != SIG:                       # resync to next local header
            chunk = f.read(1 << 20)
            idx = (head + chunk).find(SIG, 1)
            if idx < 0:
                pos += max(1, len(head) + len(chunk) - 3)
                continue
            pos += idx
            continue
        method = struct.unpack("<H", head[8:10])[0]
        csize = struct.unpack("<I", head[18:22])[0]
        fnlen = struct.unpack("<H", head[26:28])[0]
        exlen = struct.unpack("<H", head[28:30])[0]
        name = f.read(fnlen).decode("utf-8", "replace")
        f.read(exlen)
        dstart = pos + 30 + fnlen + exlen
        try:
            if method == 8:                        # deflate (streaming-safe)
                raw, consumed = inflate_stream(f, dstart)
                pos = dstart + consumed            # resync skips any data-descriptor
            elif method == 0 and csize:            # stored w/ known size
                f.seek(dstart); raw = f.read(csize); pos = dstart + csize
            else:                                   # stored streaming (unknown) -> skip
                pos = dstart
                continue
        except Exception:
            pos = dstart
            continue
        if not want(name, views, a.nframes, counts):
            continue
        op = os.path.join(a.out, name)
        os.makedirs(os.path.dirname(op), exist_ok=True)
        with open(op, "wb") as o:
            o.write(raw)
        nwrote += 1
        for v in views:
            if ("Image_subsets/%s/" % v) in name:
                counts[v] = counts.get(v, 0) + 1
    print("WROTE", nwrote, "counts", counts, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="/content/WTx")
    ap.add_argument("--views", default="C1,C2,C4,C5")
    ap.add_argument("--nframes", type=int, default=400)
    main(ap.parse_args())
