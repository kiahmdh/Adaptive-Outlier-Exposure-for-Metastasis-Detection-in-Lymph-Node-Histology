#!/usr/bin/env python
"""Phase 2 (RODEO "Adaptive Generation"), regular mode, on Camelyon16.

Faithful to RODEO's released GenerateExposures notebook. With the defaults
(--base_ckpt openai --guide clip --no_upsample) it is RODEO's generator unchanged;
only the prompts come from pathology_terms.txt:
  * inlier image -> 64px -> noised to the start step (RODEO: skip 5 of 100 -> 0.95)
  * denoised by the GLIDE base conditioned on the prompt tokens (no CFG, as in RODEO)
  * guided at every step by an image-text encoder:  mean += variance * scale * grad
      --guide clip : GLIDE's noise-aware CLIP, scale 10   (what RODEO uses)
      --guide plip : PLIP on the predicted clean image    (the swap we want to test)
      --guide none : prompt tokens only                   (ablation)
  * --upsample (default on) runs the fine-tuned 64->256 upsampler. RODEO itself just
    resized its 64px exposures; on Camelyon that would make "blurry vs sharp" the
    easiest thing for the classifier to learn. --no_upsample gives the strict version.

Scores (no filtering here; RODEO's released code does not filter) use the same encoder
as the guide. Resumable: continues after the last idx in scores.tsv.

Output: <out_dir>/images/<idx>.jpeg (or .png), <out_dir>/scores.tsv
"""
import argparse
import csv
import os
import random

import torch as th
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from glide_common import (GlideClipGuide, PlipGuide, build_glide, fill, load_openai_weights, load_prompt_lists, m11_to_uint8,
                          make_model_fn, pil_to_m11, quantize_m11, respaced_diffusion, sample_loop, save_grid,
                          seed_all, term_ensembles, tokenize)

HEADER = ("idx\tfile\tsrc\tterm\tcaption\tt_start\tguide\tsim_prompt\tsim_meta_max\t"
          "sim_normal_max\tmargin\tsim_src\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_ckpt", default="openai",
                   help="'openai' = stock GLIDE base (exactly RODEO), or a fine-tuned ema_latest.pt")
    p.add_argument("--up_ckpt", default=None,
                   help="'openai' = stock GLIDE upsampler, or a fine-tuned ema_latest.pt; needed unless --no_upsample")
    p.add_argument("--manifest", required=True, help="normal patches (source inliers)")
    p.add_argument("--prompts_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--guide", choices=["clip", "plip", "none"], default="clip")
    p.add_argument("--guide_scale", type=float, default=10.0, help="RODEO uses 10 with CLIP; sweep for PLIP")
    p.add_argument("--guide_aug", type=int, default=1, help="PLIP only: dihedral views averaged (1 = none)")
    p.add_argument("--n_images", type=int, default=20000)
    p.add_argument("--batch", type=int, default=20, help="RODEO uses 20")
    p.add_argument("--base_respacing", default="100", help="RODEO uses 100")
    p.add_argument("--t_min", type=float, default=0.95, help="start step as fraction; RODEO: 95 of 100")
    p.add_argument("--t_max", type=float, default=0.95, help="set > t_min for a random start per batch")
    p.add_argument("--cfg", type=float, default=1.0, help="classifier-free guidance; 1 = off (RODEO)")
    p.add_argument("--no_upsample", action="store_true", help="strict RODEO: keep 64px outputs")
    p.add_argument("--up_respacing", default="fast27")
    p.add_argument("--up_temp", type=float, default=0.997)
    p.add_argument("--img_format", choices=["jpeg", "png"], default="jpeg",
                   help="match the real patches (Camelyon patches are JPEG) so compression isn't a shortcut")
    p.add_argument("--jpeg_quality", type=int, default=90)
    p.add_argument("--glide_cache", default=os.path.expanduser("~/glide_model_cache"))
    p.add_argument("--plip_name", default="vinid/plip")
    p.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="one batch -> <out_dir>/dry_run_grid.png only")
    args = p.parse_args(argv)
    if not args.no_upsample and not args.up_ckpt:
        p.error("--up_ckpt is required unless --no_upsample")
    return args


def load_ema(stage, path, dev, cache_dir):
    model, _, opts = build_glide(stage)
    if path == "openai":
        load_openai_weights(model, stage, cache_dir)
    else:
        model.load_state_dict(th.load(path, map_location="cpu"))
    return model.to(dev).eval().requires_grad_(False), opts


def main(argv=None):
    args = parse_args(argv)
    dev = th.device(args.device)
    amp_dtype = th.bfloat16 if args.amp == "bf16" else None
    prompts = load_prompt_lists(args.prompts_file)
    meta_terms, normal_terms, templates = (prompts["metastasis_text_prompts"],
                                           prompts["normal_text_prompts"], prompts["templates"])
    with open(args.manifest) as f:
        src_paths = [r["path"] for r in csv.DictReader(f, delimiter="\t")]

    scores_path = os.path.join(args.out_dir, "scores.tsv")
    img_dir = os.path.join(args.out_dir, "images")
    start_idx = 0
    if not args.dry_run:
        os.makedirs(img_dir, exist_ok=True)
        if os.path.exists(scores_path) and not args.overwrite:
            with open(scores_path) as f:
                start_idx = max(0, sum(1 for _ in f) - 1)
        else:
            with open(scores_path, "w") as f:
                f.write(HEADER)

    print("=" * 70)
    print("generate_outliers.py")
    for k, v in sorted(vars(args).items()):
        print(f"  {k:16s} {v}")
    print(f"  metastasis terms {len(meta_terms)} (deduplicated)   templates {len(templates)}")
    print(f"  source patches   {len(src_paths)}")
    print(f"  resume from idx  {start_idx}")
    print("=" * 70, flush=True)
    if start_idx >= args.n_images:
        print("already complete.")
        return

    base, bopts = load_ema("base", args.base_ckpt, dev, args.glide_cache)
    bdiff = respaced_diffusion(bopts, args.base_respacing)
    if not args.no_upsample:
        up, uopts = load_ema("upsample", args.up_ckpt, dev, args.glide_cache)
        udiff = respaced_diffusion(uopts, args.up_respacing)
    if args.guide == "clip":
        guide = GlideClipGuide(dev, args.glide_cache)
    elif args.guide == "plip":
        guide = PlipGuide(dev, args.plip_name, n_aug=args.guide_aug)
    else:
        guide = None
    if guide is not None:
        M = term_ensembles(guide, meta_terms, templates)    # for scores only
        N = term_ensembles(guide, normal_terms, templates)

    n_steps = bdiff.num_timesteps
    lo, hi = max(1, round(args.t_min * n_steps)), max(1, round(args.t_max * n_steps))
    batch_ids = [0] if args.dry_run else range(start_idx // args.batch,
                                               (args.n_images + args.batch - 1) // args.batch)
    pbar = tqdm(batch_ids, desc=f"gen[{args.guide}]", unit="batch", dynamic_ncols=True,
                mininterval=5, disable=args.dry_run)
    for b in pbar:
        seed_all(args.seed * 1_000_003 + b)  # per-batch seed: a resumed run reproduces the stream
        idx0 = b * args.batch
        B = min(args.batch, args.n_images - idx0)
        srcs = random.sample(src_paths, B)
        term_ids = [random.randrange(len(meta_terms)) for _ in range(B)]
        caps = [fill(random.choice(templates), meta_terms[j]) for j in term_ids]
        t_start = random.randint(lo, hi)

        src = [Image.open(p).convert("RGB") for p in srcs]
        src256 = th.stack([pil_to_m11(im.resize((256, 256), Image.LANCZOS)) for im in src]).to(dev)
        src64 = th.stack([pil_to_m11(im.resize((64, 64), Image.BOX)) for im in src]).to(dev)

        # ---- base: noised inlier -> denoise with prompt tokens + encoder guidance
        tok, msk = tokenize(base, caps, bopts["text_ctx"], dev)
        utok, umsk = tokenize(base, [""] * B, bopts["text_ctx"], dev)
        fn = make_model_fn(base, tok, msk, utok, umsk, cfg_scale=args.cfg, amp_dtype=amp_dtype)
        if guide is not None:
            guide.set_prompts(caps)
        t_vec = th.full((B,), t_start - 1, device=dev, dtype=th.long)
        x = bdiff.q_sample(src64, t_vec, noise=th.randn_like(src64))
        low = sample_loop(bdiff, fn, x, t_start, guide=guide, guide_scale=args.guide_scale)

        # ---- optional upsampler 64 -> 256 (same prompt, no CFG, as in the GLIDE release)
        if args.no_upsample:
            final = low
        else:
            utok2, umsk2 = tokenize(up, caps, uopts["text_ctx"], dev)
            ufn = make_model_fn(up, utok2, umsk2, low_res=quantize_m11(low), amp_dtype=amp_dtype)
            xu = th.randn(B, 3, 256, 256, device=dev) * args.up_temp
            final = sample_loop(udiff, ufn, xu, udiff.num_timesteps)

        # ---- similarity scores with the guiding encoder (none for --guide none)
        nan = th.full((B,), float("nan"), device=dev)
        s_prompt = s_meta = s_norm = s_src = nan
        if guide is not None:
            with th.no_grad():
                e = guide.embed_images(final)
                e_src = guide.embed_images(src256 if not args.no_upsample else src64)
                s_prompt = (e * guide.z_t).sum(-1)
                s_meta = (e @ M.T).max(1).values
                s_norm = (e @ N.T).max(1).values
                s_src = (e * e_src).sum(-1)

        if args.dry_run:
            os.makedirs(args.out_dir, exist_ok=True)
            path = os.path.join(args.out_dir, "dry_run_grid.png")
            save_grid([src256, F.interpolate(final.float(), size=256, mode="bicubic").clamp(-1, 1)], path)
            for c, a, m, n, r in zip(caps, s_prompt.tolist(), s_meta.tolist(), s_norm.tolist(), s_src.tolist()):
                print(f"[dry] prompt {a:.3f} meta {m:.3f} normal {n:.3f} src {r:.3f} | {c}")
            print(f"[dry] mean src-sim {s_src.mean():.3f}  (image similarity to its source patch)")
            print(f"[dry] guide={args.guide} scale={args.guide_scale} t_start={t_start}/{n_steps}; "
                  f"grid (top: source, bottom: generated) -> {path}")
            return

        imgs = m11_to_uint8(final)
        rows = []
        for i in range(B):
            idx = idx0 + i
            if idx < start_idx:
                continue
            fname = os.path.join(img_dir, f"{idx:07d}.{'jpeg' if args.img_format == 'jpeg' else 'png'}")
            if args.img_format == "jpeg":
                Image.fromarray(imgs[i]).save(fname, quality=args.jpeg_quality)
            else:
                Image.fromarray(imgs[i]).save(fname, compress_level=1)
            rows.append(f"{idx}\t{fname}\t{srcs[i]}\t{meta_terms[term_ids[i]]}\t{caps[i]}\t{t_start}\t"
                        f"{args.guide}\t{s_prompt[i]:.4f}\t{s_meta[i]:.4f}\t{s_norm[i]:.4f}\t"
                        f"{(s_meta[i] - s_norm[i]):.4f}\t{s_src[i]:.4f}\n")
        with open(scores_path, "a") as f:
            f.writelines(rows)
        pbar.set_postfix(imgs=f"{idx0 + B}/{args.n_images}", t=t_start,
                         margin=f"{(s_meta - s_norm).mean():.3f}", src_sim=f"{s_src.mean():.3f}")


if __name__ == "__main__":
    main()