#!/usr/bin/env python
"""Step 2: give every normal training patch a caption for GLIDE fine-tuning.

Camelyon has no captions. We retrieve, with PLIP, the top-k terms from
normal_text_prompts for each patch. At training time a random template from the
same file wraps them ("an H&E image showing <t1> and <t2>."). This grounds
GLIDE's own text encoder in histology vocabulary so the metastasis prompts used
later are not pure gibberish to it. Only normal terms are used here.

Resumable: paths already present in --out are skipped.
Output TSV: path  terms(|-joined)  sims(|-joined)
"""
import argparse
import csv
import os
import sys

import torch as th
from PIL import Image

from glide_common import PLIP, load_prompt_lists, pil_to_m11


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--prompts_file", required=True, help="pathology_terms.txt")
    p.add_argument("--out", required=True, help="captions.tsv")
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--plip_name", default="vinid/plip")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="caption the first batch and print it; write nothing")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if os.path.exists(args.out) and args.overwrite and not args.dry_run:
        os.remove(args.out)
    prompts = load_prompt_lists(args.prompts_file)
    terms, templates = prompts["normal_text_prompts"], prompts["templates"]
    with open(args.manifest) as f:
        paths = [r["path"] for r in csv.DictReader(f, delimiter="\t")]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {line.split("\t", 1)[0] for line in f if line.strip()}
    todo = [p for p in paths if p not in done]

    print("=" * 70)
    print("caption_patches.py")
    for k, v in sorted(vars(args).items()):
        print(f"  {k:14s} {v}")
    print(f"  normal terms   {len(terms)}   templates {len(templates)}")
    print(f"  patches        {len(paths)} total, {len(done)} done, {len(todo)} to do")
    print("=" * 70, flush=True)
    if not todo:
        return

    dev = th.device(args.device)
    plip = PLIP(dev, args.plip_name)
    T = plip.term_ensembles(terms, templates)  # (n_terms, d)

    if args.dry_run:
        todo = todo[: min(len(todo), 16)]
    out = None if args.dry_run else open(args.out, "a")
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        x = th.stack([pil_to_m11(Image.open(p)) for p in chunk]).to(dev)
        with th.no_grad():
            sims = plip.image(x) @ T.T
        val, idx = sims.topk(args.topk, dim=1)
        for p, v, j in zip(chunk, val.tolist(), idx.tolist()):
            line = f"{p}\t{'|'.join(terms[k] for k in j)}\t{'|'.join(f'{s:.4f}' for s in v)}\n"
            if out is None:
                print("[dry]", line.strip())
            else:
                out.write(line)
        if out is not None:
            out.flush()
            print(f"{min(i + args.batch, len(todo))}/{len(todo)}", flush=True)
    if out is not None:
        out.close()


if __name__ == "__main__":
    sys.exit(main())
