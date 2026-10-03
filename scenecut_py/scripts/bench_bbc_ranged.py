#!/usr/bin/env python3
"""BBC Planet Earth external benchmark — STREAMED from the Zenodo zip.

Disk is only ~10GB total; the corpus zip is 4.7GB + 4.7GB extracted, which
does not fit. This script benchmarks EPISODE BY EPISODE:

  1. Opens videos.zip over HTTP Range requests (zipfile over a seekable
     RawIOBase; the central directory lives at the zip tail).
  2. Streams one bbc_NN.mp4 at a time to datasets/bbc/ (peak disk = 1 episode).
  3. Runs run_bench('bbc', config) on the episodes present on disk
     (reuses the load_bbc annotation parsing).
  4. Deletes the episode, repeats; micro-aggregates TP/FP/FN across episodes.

Usage (from repo root):
  python3 scenecut_py/scripts/bench_bbc_ranged.py [--config default] [--neural] \
      [--episodes 01,02] [--keep-last]

Output: bench/bbc_external_<config>.json (+ stdout summary).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scenecut_py"))

BBC_URL = "https://zenodo.org/records/14873790/files/videos.zip"
BBC_DIR = REPO_ROOT / "datasets" / "bbc"


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (single object)."""

    def __init__(self, url: str, timeout: int = 300):
        super().__init__()
        self.url = url
        self.timeout = timeout
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            length = r.headers.get("Content-Length")
            accept = r.headers.get("Accept-Ranges", "")
        if not length:
            raise RuntimeError(f"no Content-Length for {url}")
        if "bytes" not in accept.lower():
            # Zenodo serves ranges via wget -c resume — verify empirically.
            print(f"NOTE: Accept-Ranges={accept!r} — trying range anyway")
        self._size = int(length)

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self._size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if self.closed:
            return b""
        if n is None or n < 0:
            n = self._size - self.pos
        if n == 0 or self.pos >= self._size:
            return b""
        end = min(self.pos + n - 1, self._size - 1)
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        for attempt in (1, 2, 3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = r.read()
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"  range read retry {attempt} ({e})")
                time.sleep(3 * attempt)
        self.pos += len(data)
        return data


def stream_episode(zf: zipfile.ZipFile, name: str, dest: Path,
                   chunk: int = 8 << 20) -> Path:
    """Stream one zip entry to dest (peak memory = chunk)."""
    total = 0
    t0 = time.time()
    with zf.open(name) as src, open(dest, "wb") as out:
        while True:
            buf = src.read(chunk)
            if not buf:
                break
            out.write(buf)
            total += len(buf)
    dt = time.time() - t0
    print(f"  streamed {name} -> {dest.name}: {total/1e6:.0f} MB in {dt:.0f}s "
          f"({total/1e6/max(dt,0.01):.1f} MB/s)")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default",
                    choices=["default", "conservative", "sensitive"])
    ap.add_argument("--neural", action="store_true")
    ap.add_argument("--episodes", default=None,
                    help="comma list of episode numbers ('01,02'); default all")
    ap.add_argument("--keep-last", action="store_true",
                    help="keep the final episode file on disk (debug)")
    args = ap.parse_args()

    from scenecut.bench import run_bench
    from scenecut.bench_datasets import load_bbc

    wanted = None
    if args.episodes:
        wanted = {f"bbc_{e.strip().zfill(2)}.mp4" for e in args.episodes.split(",")}

    rf = HTTPRangeFile(BBC_URL)
    print(f"zip size: {rf._size/1e9:.2f} GB")
    zf = zipfile.ZipFile(rf)
    names = [n for n in zf.namelist() if n.endswith(".mp4")]
    names = [n for n in sorted(names) if wanted is None or Path(n).name in wanted]
    print(f"episodes to bench: {[Path(n).name for n in names]}")

    params = {"neural": True} if args.neural else {}
    rows = []
    skipped = []
    for k, name in enumerate(names):
        base = Path(name).name
        print(f"\n[{k+1}/{len(names)}] {base}")
        dest = BBC_DIR / base
        try:
            stream_episode(zf, name, dest)
            entries = load_bbc(BBC_DIR)
            if not entries:
                raise RuntimeError("load_bbc found no entries")
            rep = run_bench("bbc", args.config, params, entries=entries)
            ds = rep["datasets"]["bbc"]
            rows.extend(ds["per_video"])
            skipped.extend(ds["meta"].get("skipped_videos", [])
                           if isinstance(ds["meta"].get("skipped_videos", []), list)
                           else [])
        except Exception as e:
            print(f"  ERROR on {base}: {type(e).__name__}: {e}")
            skipped.append({"video": base, "reason": str(e)[:200]})
        finally:
            if not (args.keep_last and k == len(names) - 1):
                dest.unlink(missing_ok=True)

    # Micro-aggregate across episodes.
    def _agg(tolerance: int) -> dict:
        tp = fp = fn = 0
        off_sum = 0.0
        n_match = 0
        for r in rows:
            hc = r["hard_cuts"][f"tol{tolerance}"]
            tp += hc["tp"]
            fp += hc["fp"]
            fn += hc["fn"]
            off_sum += hc.get("mean_abs_offset", 0.0) * hc["tp"]
            n_match += hc["tp"]
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4),
                "recall": round(rc, 4), "f1": round(f1, 4),
                "mean_abs_offset": round(off_sum / n_match, 4) if n_match else None}

    out = {
        "dataset": "bbc_planet_earth",
        "source": "zenodo:14873790 (fixed.zip annotations; videos streamed ranged)",
        "config": args.config,
        "params": params,
        "tolerances": [0, 1],
        "n_videos": len(rows),
        "skipped": skipped,
        "aggregate": {"hard_cuts@tol0": _agg(0), "hard_cuts@tol1": _agg(1)},
        "per_video": rows,
    }
    suffix = ("neural" if args.neural else args.config)
    out_path = REPO_ROOT / "bench" / f"bbc_external_{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n=== BBC PE ({args.config}{' +neural' if args.neural else ''}) ===")
    print(f"videos: {len(rows)}  skipped: {len(skipped)}")
    for t in (0, 1):
        a = out["aggregate"][f"hard_cuts@tol{t}"]
        print(f"tol{t}: P={a['precision']} R={a['recall']} F1={a['f1']} "
              f"(tp {a['tp']} / fp {a['fp']} / fn {a['fn']})")
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
