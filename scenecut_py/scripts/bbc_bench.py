#!/usr/bin/env python3
"""BBC PE external benchmark — resumable chunked fetch + accumulating bench.

The sandbox kills background processes and caps Bash calls at 10 min, so this
is driven as repeated foreground invocations:

  fetch <NN>            resumable: curl-chunk the DEFLATED entry range from
                        the Zenodo zip (state in datasets/bbc/<NN>.state.json),
                        then inflate locally -> datasets/bbc/bbc_NN.mp4.
  bench [--config C] [--neural]
                        bench every bbc_NN.mp4 currently on disk; MERGE
                        per-video rows into bench/bbc_external_<suffix>.json
                        (micro-aggregated fresh from all accumulated rows).

Usage: python3 scenecut_py/scripts/bbc_bench.py index
       python3 scenecut_py/scripts/bbc_bench.py fetch 01
       python3 scenecut_py/scripts/bbc_bench.py bench --config default
       python3 scenecut_py/scripts/bbc_bench.py bench --neural
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scenecut_py"))

BBC_URL = "https://zenodo.org/records/14873790/files/videos.zip"
BBC_DIR = REPO_ROOT / "datasets" / "bbc"
CHUNK_BYTES = 96 * 1024 * 1024      # per curl invocation (~75s at 1.3MB/s)
CHUNK_TIME_BUDGET = 520             # seconds of curl per invocation


def _entry(ep: str) -> dict:
    idx = json.loads((BBC_DIR / "zip_index.json").read_text())
    key = f"videos/bbc_{ep}.mp4"
    if key not in idx:
        raise SystemExit(f"no zip index entry for {key}")
    return idx[key]


def _curl_range(url: str, start: int, end: int) -> bytes:
    """One ranged GET (curl transport — ~1.3 MB/s vs urllib ~70 KB/s)."""
    proc = subprocess.run(
        ["curl", "-s", "--max-time", "120", "--retry", "2",
         "-r", f"{start}-{end}", url],
        capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(f"curl exit {proc.returncode} fetching {start}-{end}")
    return proc.stdout


def cmd_index() -> int:
    """Rebuild datasets/bbc/zip_index.json from the REMOTE central directory.

    The index is a local artifact (gitignored) — rebuild it on fresh sandboxes
    BEFORE any fetch. Parses EOCD → central directory → per-entry LOCAL header
    (the local extra-field length can differ from the central one, so the data
    offset must come from the local header, not the central record).
    Also extracts fixed.zip (annotations) when missing.
    """
    BBC_DIR.mkdir(parents=True, exist_ok=True)

    # --- annotations (26 KB — plain GET, cheap)
    if not list(BBC_DIR.glob("fixed/*-scenes.txt")):
        fixed_zip = BBC_DIR / "fixed.zip"
        subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(fixed_zip),
                        "https://zenodo.org/records/14873790/files/fixed.zip"],
                       check=True)
        import zipfile
        with zipfile.ZipFile(fixed_zip) as zf:
            zf.extractall(BBC_DIR)
        fixed_zip.unlink(missing_ok=True)
        print("annotations extracted (fixed/)")

    # --- remote size via HEAD (follow redirects: zenodo → s3)
    head = subprocess.run(["curl", "-sIL", BBC_URL], capture_output=True,
                          text=True)
    size = None
    for line in head.stdout.lower().splitlines():
        if line.startswith("content-length:"):
            size = int(line.split(":", 1)[1].strip())
    if not size:
        raise SystemExit("could not determine remote zip size")
    print(f"remote videos.zip size: {size/1e6:.0f} MB")

    # --- EOCD (scan the last 64 KB backwards; ZIP64-aware: when the classic
    # EOCD holds 0xFFFFFFFF sentinels, the real values live in the ZIP64
    # EOCD record located via the PK\x06\x07 locator immediately before it).
    tail = _curl_range(BBC_URL, max(0, size - 65536), size - 1)
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0:
        raise SystemExit("EOCD signature not found in tail")
    eocd = tail[eocd_at:eocd_at + 22]
    _, _, _, _, n_entries, cd_size, cd_offset, _ = struct.unpack(
        "<IHHHHIIH", eocd)
    if n_entries == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        loc_at = tail.rfind(b"PK\x06\x07", 0, eocd_at)
        if loc_at < 0:
            raise SystemExit("ZIP64 sentinel in EOCD but no PK\\x06\\x07 locator")
        _, _, z64_eocd_off, _ = struct.unpack("<IIQI", tail[loc_at:loc_at + 20])
        z64 = _curl_range(BBC_URL, z64_eocd_off, z64_eocd_off + 55)
        if z64[:4] != b"PK\x06\x06":
            raise SystemExit("ZIP64 EOCD signature bad")
        (n_entries, cd_size, cd_offset) = struct.unpack("<QQQ", z64[32:56])
        print(f"ZIP64 archive detected: CD at {cd_offset}")
    print(f"central directory: {n_entries} entries, {cd_size} B at {cd_offset}")

    # --- central directory records
    cd = _curl_range(BBC_URL, cd_offset, cd_offset + cd_size - 1)
    entries: dict[str, dict] = {}
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos:pos + 4] != b"PK\x01\x02":
            break
        (sig, ver_made, ver_need, flags, method, mtime, mdate, crc,
         csize, usize, nlen, elen, clen, disk, iattr, eattr, lho) = \
            struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos:pos + 46])
        name = cd[pos + 46: pos + 46 + nlen].decode("utf-8", "replace")
        if name.endswith(".mp4"):
            entries[name] = {"compress_size": csize, "file_size": usize,
                             "method": method, "local_header_offset": lho}
        pos += 46 + nlen + elen + clen
    print(f"parsed {len(entries)} mp4 entries")

    # --- per-entry LOCAL header (exact data offset)
    for name, e in entries.items():
        lh = _curl_range(BBC_URL, e["local_header_offset"],
                         e["local_header_offset"] + 63)
        if lh[:4] != b"PK\x03\x04":
            raise SystemExit(f"local header bad for {name}")
        nlen, elen = struct.unpack("<HH", lh[26:30])
        e["data_offset"] = e["local_header_offset"] + 30 + nlen + elen
        del e["local_header_offset"]

    out = {k: v for k, v in sorted(entries.items())}
    (BBC_DIR / "zip_index.json").write_text(json.dumps(out, indent=1))
    total = sum(e["file_size"] for e in out.values())
    print(f"index written: {len(out)} videos, {total/1e9:.2f} GB uncompressed")
    return 0


def cmd_fetch(ep: str) -> int:
    entry = _entry(ep)
    raw = BBC_DIR / f"bbc_{ep}.mp4.deflate"
    state = BBC_DIR / f"bbc_{ep}.state.json"
    mp4 = BBC_DIR / f"bbc_{ep}.mp4"
    if mp4.exists():
        print(f"already fetched: {mp4}")
        return 0

    off = entry["data_offset"]
    csize = entry["compress_size"]
    done = 0
    if state.exists():
        st = json.loads(state.read_text())
        done = st.get("downloaded", 0)
    t0 = time.time()
    while done < csize and time.time() - t0 < CHUNK_TIME_BUDGET:
        want = min(CHUNK_BYTES, csize - done)
        start, end = off + done, off + done + want - 1
        # curl: ranged GET appended to the raw file (resume-safe: state tracks
        # the byte count actually written).
        cmd = ["curl", "-s", "--max-time", "480", "--retry", "2",
               "-r", f"{start}-{end}", BBC_URL]
        with open(raw, "ab") as f:
            proc = subprocess.run(cmd, stdout=f)
        if proc.returncode != 0:
            print(f"curl exit {proc.returncode} at {done}/{csize} — retry next invocation")
            break
        got = raw.stat().st_size
        if got < done:  # file truncated externally — restart
            print("state/file mismatch — restarting download")
            raw.unlink(missing_ok=True)
            done = 0
            continue
        done = got
        state.write_text(json.dumps({"downloaded": done, "total": csize}))
        print(f"  chunk done: {done/1e6:.0f}/{csize/1e6:.0f} MB "
              f"({done/1e6/(time.time()-t0):.1f} MB/s avg)")

    if done < csize:
        print(f"INCOMPLETE ({done}/{csize}) — run fetch {ep} again to resume")
        return 1

    # Inflate (raw deflate stream of the entry data).
    print("inflating...")
    comp = raw.read_bytes()
    d = zlib.decompressobj(-15)
    data = d.decompress(comp, entry["file_size"])
    if len(data) != entry["file_size"]:
        raw.unlink(missing_ok=True)
        state.unlink(missing_ok=True)
        raise SystemExit(f"inflate size mismatch: {len(data)} != {entry['file_size']}")
    mp4.write_bytes(data)
    raw.unlink(missing_ok=True)
    state.unlink(missing_ok=True)
    print(f"OK: {mp4} ({entry['file_size']/1e6:.0f} MB)")
    return 0


def cmd_bench(config: str, neural: bool) -> int:
    from scenecut.bench import run_bench
    from scenecut.bench_datasets import load_bbc

    videos = sorted(BBC_DIR.glob("bbc_*.mp4"))
    if not videos:
        raise SystemExit("no bbc_*.mp4 on disk — fetch episodes first")
    print(f"benching {len(videos)} episodes ({config}{' +neural' if neural else ''})")
    t0 = time.time()
    rep = run_bench("bbc", config, {"neural": True} if neural else {})
    dt = time.time() - t0
    ds = rep["datasets"]["bbc"]
    rows = ds["per_video"]
    print(f"  evaluated {len(rows)} videos in {dt:.0f}s")

    suffix = "neural" if neural else config
    out_path = REPO_ROOT / "bench" / f"bbc_external_{suffix}.json"
    prev = []
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text()).get("per_video", [])
        except Exception:
            prev = []
    by_video = {r["video"]: r for r in prev}
    for r in rows:
        by_video[r["video"]] = r          # fresh run wins per video
    all_rows = [by_video[k] for k in sorted(by_video)]

    def _agg(tolerance: int) -> dict:
        tp = fp = fn = 0
        for r in all_rows:
            hc = r["hard_cuts"][f"tol{tolerance}"]
            tp += hc["tp"]; fp += hc["fp"]; fn += hc["fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4),
                "recall": round(rc, 4), "f1": round(f1, 4)}

    out = {
        "dataset": "bbc_planet_earth",
        "source": "zenodo:14873790 (fixed.zip annotations; curl-ranged videos)",
        "config": config,
        "neural": neural,
        "tolerances": [0, 1],
        "n_videos": len(all_rows),
        "aggregate": {"hard_cuts@tol0": _agg(0), "hard_cuts@tol1": _agg(1)},
        "per_video": all_rows,
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    for t in (0, 1):
        a = out["aggregate"][f"hard_cuts@tol{t}"]
        print(f"  tol{t}: P={a['precision']} R={a['recall']} F1={a['f1']} "
              f"(tp {a['tp']} / fp {a['fp']} / fn {a['fn']})")
    print(f"report: {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="rebuild zip_index.json + annotations")
    f = sub.add_parser("fetch")
    f.add_argument("episode")
    b = sub.add_parser("bench")
    b.add_argument("--config", default="default",
                   choices=["default", "conservative", "sensitive"])
    b.add_argument("--neural", action="store_true")
    args = ap.parse_args()
    if args.cmd == "index":
        return cmd_index()
    if args.cmd == "fetch":
        return cmd_fetch(args.episode.zfill(2))
    return cmd_bench(args.config, args.neural)


if __name__ == "__main__":
    raise SystemExit(main())
