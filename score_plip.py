#!/usr/bin/env python
"""Re-score generated outliers with PLIP (post hoc, no regeneration).

generate_outliers.py scores each image with the guiding encoder (GLIDE's CLIP for the
RODEO-exact run). This writes the same columns computed with PLIP instead, in the same
row order, so train_classifier.py can filter on PLIP similarities by pointing
--outlier_scores at the new file. Works on a finished or still-running generation
(rows already scored are skipped on rerun).

Columns recomputed: sim_prompt (image vs its own caption), sim_meta_max / sim_normal_max
(best metastasis / normal term, template-ensembled), margin, sim_src (image vs its source
patch). All other columns are copied unchanged.
"""
import argparse
import csv
import os

import torch as th
from PIL import Image
from tqdm import tqdm

from glide_common import PLIP, load_prompt_lists, pil_to_m11

COLS = ["idx", "file", "src", "term", "caption", "t_start", "guide", "sim_prompt", "sim_meta_max",
        "sim_normal_max", "margin", "sim_src"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", required=True, help="gen/<run>/scores.tsv")
    p.add_argument("--prompts_file", required=True)
    p.add_argument("--out", default=None, help="default: scores_plip.tsv next to --scores")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--plip_name", default="vinid/plip")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="score one batch and print it; write nothing")
    return p.parse_args(argv)


def load_m11(path, size):
    return pil_to_m11(Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC))


def main(argv=None):
    args = parse_args(argv)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.scores)), "scores_plip.tsv")
    with open(args.scores) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    done = 0
    if os.path.exists(out) and not args.overwrite and not args.dry_run:
        with open(out) as f:
            done = max(0, sum(1 for _ in f) - 1)
    todo = rows[done:]

    print("=" * 70)
    print("score_plip.py")
    for k, v in sorted(vars(args).items()):
        print(f"  {k:12s} {v}")
    print(f"  out          {out}")
    print(f"  rows         {len(rows)} total, {done} done, {len(todo)} to do")
    print("=" * 70, flush=True)
    if not todo:
        return

    prompts = load_prompt_lists(args.prompts_file)
    dev = th.device(args.device)
    plip = PLIP(dev, args.plip_name)
    M = plip.term_ensembles(prompts["metastasis_text_prompts"], prompts["templates"])
    N = plip.term_ensembles(prompts["normal_text_prompts"], prompts["templates"])

    if args.dry_run:
        todo = todo[: min(len(todo), 16)]
    else:
        if done == 0:
            with open(out, "w") as f:
                f.write("\t".join(COLS) + "\n")
    fh = None if args.dry_run else open(out, "a")
    for i in tqdm(range(0, len(todo), args.batch), desc="plip-score", unit="batch",
                  dynamic_ncols=True, mininterval=5, disable=args.dry_run):
        chunk = todo[i:i + args.batch]
        # PLIP resizes to 224 internally; load both at 256 so they go through the same path
        x = th.stack([load_m11(r["file"], 256) for r in chunk]).to(dev)
        xs = th.stack([load_m11(r["src"], 256) for r in chunk]).to(dev)
        with th.no_grad():
            e, es = plip.image(x), plip.image(xs)
            zt = plip.text([r["caption"] for r in chunk])
            s_prompt = (e * zt).sum(-1)
            s_meta = (e @ M.T).max(1).values
            s_norm = (e @ N.T).max(1).values
            s_src = (e * es).sum(-1)
        for r, a, m, n, s in zip(chunk, s_prompt.tolist(), s_meta.tolist(), s_norm.tolist(), s_src.tolist()):
            new = dict(r, sim_prompt=f"{a:.4f}", sim_meta_max=f"{m:.4f}", sim_normal_max=f"{n:.4f}",
                       margin=f"{m - n:.4f}", sim_src=f"{s:.4f}")
            line = "\t".join(str(new[c]) for c in COLS) + "\n"
            if fh is None:
                print(f"[dry] prompt {a:.3f} meta {m:.3f} normal {n:.3f} src {s:.3f} | {r['caption']}")
            else:
                fh.write(line)
        if fh is not None:
            fh.flush()
    if fh is not None:
        fh.close()
        print(f"saved {out}")


if __name__ == "__main__":
    main()
