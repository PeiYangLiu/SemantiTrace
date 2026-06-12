#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import aiohttp
from PIL import Image, ImageOps

METADATA_URL = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Download 1M unique OpenImages thumbnails resized to 512px max side.")
    p.add_argument("--output_root", default="/datadisk/peiyang/semantitrace_openimages_1m_512")
    p.add_argument("--target", type=int, default=1_000_000)
    p.add_argument("--max_metadata_rows", type=int, default=1_750_000)
    p.add_argument("--concurrency", type=int, default=96)
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--max_side", type=int, default=512)
    p.add_argument("--jpeg_quality", type=int, default=88)
    p.add_argument("--seed", type=int, default=20260607)
    p.add_argument("--metadata_url", default=METADATA_URL)
    p.add_argument("--log_every", type=int, default=1000)
    return p.parse_args()


def image_path(images_dir: Path, idx: int, image_id: str) -> Path:
    shard = idx // 10000
    return images_dir / f"shard_{shard:04d}" / f"{idx:08d}_{image_id}.jpg"


def atomic_append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def load_done(manifest_path: Path) -> set[str]:
    done: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("ok") and row.get("image_id"):
                    done.add(str(row["image_id"]))
    return done


def read_candidates(metadata_path: Path, metadata_url: str, max_rows: int, seed: int) -> list[dict[str, str]]:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if not metadata_path.exists():
        print(f"[metadata] downloading {metadata_url} -> {metadata_path}", flush=True)
        with urlopen(metadata_url) as src, metadata_path.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    rows: list[dict[str, str]] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            url = row.get("Thumbnail300KURL") or row.get("OriginalURL")
            image_id = row.get("ImageID")
            if not url or not image_id:
                continue
            rows.append({
                "image_id": image_id,
                "url": url,
                "license": row.get("License", ""),
                "original_url": row.get("OriginalURL", ""),
                "landing_url": row.get("OriginalLandingURL", ""),
                "title": row.get("Title", ""),
                "author": row.get("Author", ""),
            })
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def resize_jpeg_bytes(data: bytes, max_side: int, quality: int) -> tuple[bytes, tuple[int, int]]:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue(), image.size


async def fetch_one(session: aiohttp.ClientSession, row: dict[str, str], idx: int, out_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"ok": True, "skipped_existing": True, "idx": idx, "image_id": row["image_id"], "path": str(out_path), **row}
    try:
        async with session.get(row["url"], timeout=args.timeout) as resp:
            if resp.status != 200:
                return {"ok": False, "idx": idx, "image_id": row["image_id"], "status": resp.status, "url": row["url"], "error": "http"}
            data = await resp.read()
        jpg, size = await asyncio.to_thread(resize_jpeg_bytes, data, args.max_side, args.jpeg_quality)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_bytes(jpg)
        os.replace(tmp, out_path)
        return {"ok": True, "idx": idx, "image_id": row["image_id"], "path": str(out_path), "width": size[0], "height": size[1], **row}
    except Exception as exc:
        return {"ok": False, "idx": idx, "image_id": row.get("image_id"), "url": row.get("url"), "error": type(exc).__name__, "message": str(exc)[:300]}


async def main_async(args: argparse.Namespace) -> None:
    out_root = Path(args.output_root)
    images_dir = out_root / "images"
    meta_dir = out_root / "metadata"
    manifest_path = meta_dir / "selected_1m_manifest.jsonl"
    failures_path = meta_dir / "download_failures.jsonl"
    progress_path = meta_dir / "progress.json"
    metadata_path = meta_dir / "openimages_train_images_boxable_with_rotation.csv"
    meta_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    done = load_done(manifest_path)
    candidates = read_candidates(metadata_path, args.metadata_url, args.max_metadata_rows, args.seed)
    print(f"[start] candidates={len(candidates)} already_done={len(done)} target={args.target}", flush=True)
    if len(done) >= args.target:
        print("[done] target already reached", flush=True)
        return

    stop = False
    def handle_signal(signum, frame):
        nonlocal stop
        stop = True
        print(f"[signal] received {signum}; will stop after in-flight batch", flush=True)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    success = len(done)
    attempted = 0
    start_time = time.time()
    connector = aiohttp.TCPConnector(limit=args.concurrency, ttl_dns_cache=300)
    headers = {"User-Agent": "Mozilla/5.0 SemantiTraceOpenImagesDownloader/1.0"}
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        pending: set[asyncio.Task] = set()
        candidate_iter = iter(enumerate(candidates))
        next_out_idx = success
        exhausted = False
        while success < args.target and not stop:
            while len(pending) < args.concurrency and not exhausted and success + len(pending) < args.target + args.concurrency * 4:
                try:
                    _, row = next(candidate_iter)
                except StopIteration:
                    exhausted = True
                    break
                if row["image_id"] in done:
                    continue
                out_idx = next_out_idx
                next_out_idx += 1
                task = asyncio.create_task(fetch_one(session, row, out_idx, image_path(images_dir, out_idx, row["image_id"]), args))
                pending.add(task)
            if not pending:
                break
            done_tasks, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done_tasks:
                result = task.result()
                attempted += 1
                if result.get("ok"):
                    success += 1
                    done.add(str(result["image_id"]))
                    atomic_append_jsonl(manifest_path, result)
                else:
                    atomic_append_jsonl(failures_path, result)
                if success % args.log_every == 0 or attempted % (args.log_every * 2) == 0:
                    elapsed = max(1e-6, time.time() - start_time)
                    progress = {
                        "success": success,
                        "attempted_since_start": attempted,
                        "target": args.target,
                        "elapsed_sec": elapsed,
                        "success_per_sec": success / elapsed,
                        "failures_path": str(failures_path),
                        "manifest_path": str(manifest_path),
                    }
                    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                    print(f"[progress] success={success} attempted={attempted} rate={success/elapsed:.2f}/s", flush=True)
        for task in pending:
            task.cancel()
    elapsed = time.time() - start_time
    final = {"success": success, "target": args.target, "elapsed_sec": elapsed, "complete": success >= args.target}
    progress_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("[final]", json.dumps(final), flush=True)


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
