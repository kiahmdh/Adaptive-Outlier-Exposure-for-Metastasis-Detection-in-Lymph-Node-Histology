#!/usr/bin/env python
"""Step 1: extract tissue patches from Camelyon16 *normal* training slides.

Normal-only by design: slides whose file name does not start with "normal" are
refused unless --allow_non_normal is passed. Resumable: a slide whose rows file
already exists is skipped. The manifest is rebuilt from finished slides at the end.

Output:
  <out_dir>/patches/<slide>/<slide>_x<X>_y<Y>.png
  <out_dir>/patches/<slide>/_rows.tsv        (per-slide done marker)
  <out_dir>/manifest.tsv                      path  slide  x0  y0  mpp
"""
import argparse
import glob
import os
import random
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np
from PIL import Image


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slide_glob", default="/home/user01/camelyon16/training/normal/*.tif")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--patch_size", type=int, default=256, help="output patch side in pixels")
    p.add_argument("--target_mpp", type=float, default=0.5, help="0.5 ~ 20x; match your DN2 setting")
    p.add_argument("--mpp0_fallback", type=float, default=0.243, help="used if a slide has no mpp metadata")
    p.add_argument("--tissue_frac", type=float, default=0.5, help="min tissue fraction per patch (mask level)")
    p.add_argument("--max_white", type=float, default=0.80, help="reject patch if >this fraction of pixels is near-white")
    p.add_argument("--max_per_slide", type=int, default=500)
    p.add_argument("--mask_downsample", type=float, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_non_normal", action="store_true", help="breaks the normal-only assumption; off by default")
    p.add_argument("--overwrite", action="store_true", help="re-extract slides that are already done")
    p.add_argument("--dry_run", action="store_true", help="report slides, mpp and candidate counts for 3 slides; write nothing")
    return p.parse_args(argv)


def otsu(values):
    hist, edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    return centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def slide_geometry(slide, args):
    import openslide
    mpp0 = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp0 = float(mpp0) if mpp0 else args.mpp0_fallback
    target_ds = args.target_mpp / mpp0
    level = slide.get_best_level_for_downsample(target_ds)
    level_ds = slide.level_downsamples[level]
    read_size = int(round(args.patch_size * target_ds / level_ds))
    footprint0 = int(round(args.patch_size * target_ds))  # patch side in level-0 pixels
    return mpp0, level, read_size, footprint0


def tissue_candidates(slide, footprint0, args):
    mlevel = slide.get_best_level_for_downsample(args.mask_downsample)
    mds = slide.level_downsamples[mlevel]
    w, h = slide.level_dimensions[mlevel]
    img = np.asarray(slide.read_region((0, 0), mlevel, (w, h)).convert("RGB"), dtype=np.float32) / 255.0
    mx, mn = img.max(2), img.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    valid = mx > 0.2  # drop black / transparent background
    thr = max(otsu(sat[valid]) if valid.any() else 0.1, 0.05)
    mask = (sat > thr) & valid & (mn < 0.9)

    W0, H0 = slide.level_dimensions[0]
    cands = []
    fp_m = max(1, int(round(footprint0 / mds)))
    for y0 in range(0, H0 - footprint0 + 1, footprint0):
        my = int(y0 / mds)
        for x0 in range(0, W0 - footprint0 + 1, footprint0):
            mxx = int(x0 / mds)
            win = mask[my:my + fp_m, mxx:mxx + fp_m]
            if win.size and win.mean() >= args.tissue_frac:
                cands.append((x0, y0))
    return cands


def process_slide(job):
    path, args = job
    import openslide
    name = os.path.splitext(os.path.basename(path))[0]
    sdir = os.path.join(args.out_dir, "patches", name)
    rows_path = os.path.join(sdir, "_rows.tsv")
    if os.path.exists(rows_path) and not args.overwrite:
        return name, -1, "skipped (done)"
    os.makedirs(sdir, exist_ok=True)
    t0 = time.time()
    slide = openslide.OpenSlide(path)
    mpp0, level, read_size, fp0 = slide_geometry(slide, args)
    cands = tissue_candidates(slide, fp0, args)
    rng = random.Random(args.seed + zlib.crc32(name.encode()))
    rng.shuffle(cands)
    rows = []
    for x0, y0 in cands:
        if len(rows) >= args.max_per_slide:
            break
        tile = slide.read_region((x0, y0), level, (read_size, read_size)).convert("RGB")
        if read_size != args.patch_size:
            tile = tile.resize((args.patch_size, args.patch_size), Image.LANCZOS)
        a = np.asarray(tile, dtype=np.float32) / 255.0
        if (a.min(2) > 0.85).mean() > args.max_white:
            continue
        fname = os.path.join(sdir, f"{name}_x{x0}_y{y0}.png")
        tile.save(fname, compress_level=1)
        rows.append(f"{fname}\t{name}\t{x0}\t{y0}\t{mpp0:.4f}\n")
    tmp = rows_path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(rows)
    os.replace(tmp, rows_path)
    return name, len(rows), f"{len(cands)} cands, level {level}, read {read_size}px, {time.time() - t0:.0f}s"


def main(argv=None):
    args = parse_args(argv)
    args.out_dir = os.path.abspath(args.out_dir)  # manifest stores absolute paths
    slides = sorted(glob.glob(args.slide_glob))
    if not slides:
        sys.exit(f"No slides match {args.slide_glob}")
    bad = [s for s in slides if not os.path.basename(s).lower().startswith("normal")]
    if bad and not args.allow_non_normal:
        sys.exit(f"{len(bad)} non-normal slides matched (e.g. {os.path.basename(bad[0])}). "
                 "Training must stay normal-only; fix --slide_glob or pass --allow_non_normal.")

    print("=" * 70)
    print("extract_patches.py")
    for k, v in sorted(vars(args).items()):
        print(f"  {k:18s} {v}")
    print(f"  {'n_slides':18s} {len(slides)}")
    print("=" * 70, flush=True)

    if args.dry_run:
        import openslide
        for s in slides[:3]:
            sl = openslide.OpenSlide(s)
            mpp0, level, rs, fp0 = slide_geometry(sl, args)
            n = len(tissue_candidates(sl, fp0, args))
            print(f"[dry] {os.path.basename(s)}: mpp0={mpp0:.4f} level={level} read={rs}px "
                  f"footprint0={fp0}px candidates={n}")
        print("[dry] nothing written.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    jobs = [(s, args) for s in slides]
    with Pool(args.workers) as pool:
        for i, (name, n, msg) in enumerate(pool.imap_unordered(process_slide, jobs), 1):
            print(f"[{i}/{len(jobs)}] {name}: {n if n >= 0 else '-'} patches  ({msg})", flush=True)

    manifest = os.path.join(args.out_dir, "manifest.tsv")
    total = 0
    with open(manifest + ".tmp", "w") as out:
        out.write("path\tslide\tx0\ty0\tmpp0\n")
        for s in slides:
            name = os.path.splitext(os.path.basename(s))[0]
            rp = os.path.join(args.out_dir, "patches", name, "_rows.tsv")
            if os.path.exists(rp):
                lines = open(rp).readlines()
                out.writelines(lines)
                total += len(lines)
    os.replace(manifest + ".tmp", manifest)
    print(f"manifest: {manifest}  ({total} patches from {len(slides)} slides)")


if __name__ == "__main__":
    main()
