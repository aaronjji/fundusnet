"""
presize_images.py — Resize all fundus images to 512×512 JPEG.

Reads from data/train/ (4752×3168 originals, ~1.5 MB each).
Writes to  data/train_512/ (~30 KB each).

DataLoader batch time: ~3.5s → ~0.3s after caching.

Usage:
    python scripts/presize_images.py
    python scripts/presize_images.py --workers 8 --quality 92
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm import tqdm


def resize_one(src: Path, dst: Path, size: int, quality: int) -> str:
    if dst.exists():
        return "skip"
    img = cv2.imread(str(src))
    if img is None:
        return f"fail:{src.name}"
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "ok"


def main(args):
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpeg", ".jpg", ".png"}
    sources = [p for p in src_dir.iterdir() if p.suffix.lower() in exts]
    print(f"Found {len(sources):,} images in {src_dir}")
    print(f"Output → {dst_dir}  (size={args.size}, quality={args.quality}, workers={args.workers})")

    ok = skip = fail = 0
    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for src in sources:
            dst = dst_dir / (src.stem + ".jpeg")
            fut = pool.submit(resize_one, src, dst, args.size, args.quality)
            futures[fut] = src.name

        with tqdm(total=len(futures), unit="img") as bar:
            for fut in as_completed(futures):
                result = fut.result()
                if result == "ok":
                    ok += 1
                elif result == "skip":
                    skip += 1
                else:
                    fail += 1
                bar.set_postfix(ok=ok, skip=skip, fail=fail)
                bar.update(1)

    print(f"\nDone: {ok} written, {skip} skipped, {fail} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir",  default="data/train")
    parser.add_argument("--dst_dir",  default="data/train_512")
    parser.add_argument("--size",     type=int, default=512)
    parser.add_argument("--quality",  type=int, default=92)
    parser.add_argument("--workers",  type=int, default=8)
    main(parser.parse_args())
