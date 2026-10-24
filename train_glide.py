#!/usr/bin/env python
"""Step 3: fine-tune OpenAI GLIDE (filtered) on Camelyon16 normal patches.

Two stages, run separately (they are independent and can run back to back):
  --stage base      text-conditional 64x64 model       (init: OpenAI "base")
  --stage upsample  text-conditional 64->256 upsampler (init: OpenAI "upsample")

Loss is the simple epsilon MSE (as in glide-finetune); the learned-variance head
is kept from the OpenAI checkpoint. Captions are dropped with prob --p_uncond so
classifier-free guidance keeps working.

Resume-after-crash: if <out_dir>/ckpt_latest.pt exists it is resumed unless
--overwrite is given. Files written:
  ckpt_latest.pt       model + ema + optimiser + step   (for resuming)
  ema_latest.pt        EMA weights only                 (for generation)
  ema_step<N>.pt       EMA snapshot every --keep_every steps
  log.tsv              step  loss  lr  sec_per_step
  samples/step<N>.png  preview grids
"""
import argparse
import copy
import csv
import os
import random
import sys
import time

import torch as th
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from glide_common import (build_glide, fill, load_openai_weights, load_prompt_lists, make_model_fn,
                          pil_to_m11, quantize_m11, respaced_diffusion, sample_loop, save_grid, seed_all,
                          tokenize)

STAGE_DEFAULTS = {"base": dict(batch=8, accum=4), "upsample": dict(batch=2, accum=4)}  # fits 24GB; eff. batch 32 / 8


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["base", "upsample"], required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--captions", default=None, help="captions.tsv from caption_patches.py")
    p.add_argument("--caption_mode", choices=["constant", "plip"], default="constant",
                   help="constant keeps PLIP out of fine-tuning, so both guidance arms share one GLIDE")
    p.add_argument("--constant_caption", default="an H&E image of lymph node tissue.")
    p.add_argument("--prompts_file", required=True, help="pathology_terms.txt (templates + preview prompts)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--glide_cache", default=os.path.expanduser("~/glide_model_cache"))
    p.add_argument("--steps", type=int, default=60000, help="optimiser steps")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.9995)
    p.add_argument("--p_uncond", type=float, default=0.2)
    p.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    p.add_argument("--max_patches", type=int, default=0, help="random subset for pilots (0 = all)")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--keep_every", type=int, default=10000)
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true", help="ignore an existing ckpt_latest.pt and start fresh")
    p.add_argument("--dry_run", action="store_true", help="build everything, run 2 steps, write nothing")
    args = p.parse_args(argv)
    for k, v in STAGE_DEFAULTS[args.stage].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.caption_mode == "plip" and not args.captions:
        p.error("--caption_mode plip needs --captions (or use --caption_mode constant)")
    return args


class PatchDataset(Dataset):
    def __init__(self, paths, captions, templates, stage, caption_mode, constant_caption, p_uncond):
        self.paths, self.captions, self.templates = paths, captions, templates
        self.stage, self.mode, self.const, self.p_uncond = stage, caption_mode, constant_caption, p_uncond

    def __len__(self):
        return len(self.paths)

    def caption(self, path):
        if random.random() < self.p_uncond:
            return ""
        if self.mode == "constant":
            return self.const
        terms = self.captions[path]
        return fill(random.choice(self.templates), " and ".join(terms))

    def __getitem__(self, i):
        path = self.paths[i]
        img = Image.open(path).convert("RGB")
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.LANCZOS)
        # histology is orientation-free: random dihedral augmentation
        k = random.randint(0, 3)
        if k:
            img = img.rotate(90 * k)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        low = img.resize((64, 64), Image.BOX)
        cap = self.caption(path)
        if self.stage == "base":
            return pil_to_m11(low), pil_to_m11(low), cap  # (target, unused, caption)
        return pil_to_m11(img), pil_to_m11(low), cap      # (target 256, low_res 64, caption)


def infinite(loader):
    while True:
        for b in loader:
            yield b


def banner(args, n):
    print("=" * 70)
    print(f"train_glide.py  stage={args.stage}")
    for k, v in sorted(vars(args).items()):
        print(f"  {k:16s} {v}")
    print(f"  {'n_patches':16s} {n}")
    print(f"  {'eff_batch':16s} {args.batch * args.accum}")
    print("=" * 70, flush=True)


@th.no_grad()
def preview(args, ema, opts, prompts, fixed, step, dev, amp_dtype):
    ema.eval()
    rng_state = th.get_rng_state(), th.cuda.get_rng_state_all() if dev.type == "cuda" else None
    th.manual_seed(1234)
    n = 8
    if args.stage == "base":
        diff = respaced_diffusion(opts, "50")
        normal = [fill("an H&E image showing CLASSNAME.", t) for t in prompts["normal_text_prompts"][:n]]
        meta = [fill("an H&E image showing CLASSNAME.", t) for t in prompts["metastasis_text_prompts"][:n]]
        rows = []
        for caps in (normal, meta):
            tok, msk = tokenize(ema, caps, opts["text_ctx"], dev)
            utok, umsk = tokenize(ema, [""] * n, opts["text_ctx"], dev)
            fn = make_model_fn(ema, tok, msk, utok, umsk, cfg_scale=3.0, amp_dtype=amp_dtype)
            x = th.randn(n, 3, 64, 64, device=dev)
            rows.append(sample_loop(diff, fn, x, diff.num_timesteps))
    else:
        diff = respaced_diffusion(opts, "fast27")
        hr, lr, caps = fixed
        hr, lr = hr[:n].to(dev), lr[:n].to(dev)
        tok, msk = tokenize(ema, list(caps)[:n], opts["text_ctx"], dev)
        fn = make_model_fn(ema, tok, msk, low_res=quantize_m11(lr), amp_dtype=amp_dtype)
        x = th.randn(len(lr), 3, 256, 256, device=dev) * 0.997
        up = sample_loop(diff, fn, x, diff.num_timesteps)
        rows = [F.interpolate(lr, size=256, mode="bicubic", align_corners=False).clamp(-1, 1), up, hr]
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
    save_grid(rows, os.path.join(args.out_dir, "samples", f"step{step:07d}.png"))
    th.set_rng_state(rng_state[0])
    if rng_state[1] is not None:
        th.cuda.set_rng_state_all(rng_state[1])


def atomic_save(obj, path):
    th.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def main(argv=None):
    args = parse_args(argv)
    seed_all(args.seed)
    dev = th.device(args.device)
    amp_dtype = th.bfloat16 if args.amp == "bf16" else None
    prompts = load_prompt_lists(args.prompts_file)

    with open(args.manifest) as f:
        paths = [r["path"] for r in csv.DictReader(f, delimiter="\t")]
    captions = {}
    if args.caption_mode == "plip":
        with open(args.captions) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    captions[parts[0]] = parts[1].split("|")
        missing = sum(p not in captions for p in paths)
        if missing:
            sys.exit(f"{missing} manifest patches have no caption; finish caption_patches.py first.")
    if args.max_patches and args.max_patches < len(paths):
        paths = random.Random(args.seed).sample(paths, args.max_patches)  # random, not first-N

    banner(args, len(paths))
    ckpt_path = os.path.join(args.out_dir, "ckpt_latest.pt")
    resuming = os.path.exists(ckpt_path) and not args.overwrite and not args.dry_run
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)

    model, diffusion, opts = build_glide(args.stage)
    if resuming:
        ck = th.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"])
    else:
        load_openai_weights(model, args.stage, args.glide_cache)
    model.to(dev).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step = 0
    if resuming:
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        del ck
        print(f"resumed from {ckpt_path} at step {step}", flush=True)

    ds = PatchDataset(paths, captions, prompts["templates"], args.stage, args.caption_mode,
                      args.constant_caption, args.p_uncond)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=True, pin_memory=True, persistent_workers=args.workers > 0,
                        worker_init_fn=lambda w: random.seed(args.seed * 1000 + step + w))
    it = infinite(loader)
    fixed = next(iter(DataLoader(ds, batch_size=8, shuffle=False)))  # preview batch

    log_path = os.path.join(args.out_dir, "log.tsv")
    if not args.dry_run and not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("step\tloss\tlr\tsec_per_step\n")

    T = diffusion.num_timesteps
    run_loss, t_last = 0.0, time.time()
    end = step + 2 if args.dry_run else args.steps
    while step < end:
        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.accum):
            target, low, caps = next(it)
            target = target.to(dev, non_blocking=True)
            tok, msk = tokenize(model, list(caps), opts["text_ctx"], dev)
            t = th.randint(0, T, (target.shape[0],), device=dev)
            noise = th.randn_like(target)
            x_t = diffusion.q_sample(target, t, noise=noise)
            extra = {"low_res": low.to(dev, non_blocking=True)} if args.stage == "upsample" else {}
            with th.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None and dev.type == "cuda"):
                out = model(x_t, t, tokens=tok, mask=msk, **extra)
            loss = F.mse_loss(out[:, :3].float(), noise) / args.accum
            loss.backward()
            acc += loss.item()
        th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        with th.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
        step += 1
        run_loss += acc

        if args.dry_run:
            print(f"[dry] step {step} loss {acc:.4f}", flush=True)
            continue
        if step % args.log_every == 0:
            sps = (time.time() - t_last) / args.log_every
            with open(log_path, "a") as f:
                f.write(f"{step}\t{run_loss / args.log_every:.5f}\t{lr:.2e}\t{sps:.3f}\n")
            print(f"step {step:7d}  loss {run_loss / args.log_every:.5f}  lr {lr:.2e}  "
                  f"{sps:.2f}s/step  eta {sps * (args.steps - step) / 3600:.1f}h", flush=True)
            run_loss, t_last = 0.0, time.time()
        if step % args.sample_every == 0 or step == args.steps:
            preview(args, ema, opts, prompts, fixed, step, dev, amp_dtype)
        if step % args.save_every == 0 or step == args.steps:
            atomic_save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                         "step": step, "args": vars(args), "opts": opts}, ckpt_path)
            atomic_save(ema.state_dict(), os.path.join(args.out_dir, "ema_latest.pt"))
            if step % args.keep_every == 0:
                atomic_save(ema.state_dict(), os.path.join(args.out_dir, f"ema_step{step:07d}.pt"))
            print(f"saved step {step}", flush=True)

    if args.dry_run:
        print("[dry] ok, nothing written.")


if __name__ == "__main__":
    main()