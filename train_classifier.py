#!/usr/bin/env python
"""Phases 3-4 (regular mode): normal-vs-synthetic-outlier classifier on frozen
Prov-GigaPath features, then scoring of test patches.

RODEO's classifier recipe minus PGD: 2-way softmax head, cross-entropy, Adam lr 1e-4
wd 5e-5. Class ratio: by default 2 normals per kept (filtered) outlier (--normal_ratio 2);
--normal_ratio 0 restores RODEO's 1:1 resampling of outliers against all normals.
OOD score = softmax probability of the "outlier" class.

Validation: 10% of normal *slides* are held out, together with every synthetic image
whose source patch came from those slides (no leakage through the source image).
Val AUROC (held-out normal vs held-out synthetic) is a sanity check only; the real
number is test AUROC on Camelyon16.

Output (<out_dir>):
  head.pt            head weights + feature mean/std
  test_scores.npy    P(outlier) per test row, same order as --test_feats
  summary.tsv        config + train/val/test metrics (test AUROC if --test_labels given)
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--normal_feats", required=True, help="feats/normal_train.npy")
    p.add_argument("--normal_manifest", required=True, help="manifest.tsv (same row order as normal_feats)")
    p.add_argument("--normal_ok", default=None,
                   help="optional bool .npy (e.g. DN2 bank ok.npy); rows marked False are dropped")
    p.add_argument("--outlier_feats", required=True, help="feats/<gen run>.npy")
    p.add_argument("--outlier_scores", required=True, help="gen/<run>/scores.tsv (same row order)")
    p.add_argument("--max_normal_sim", type=float, default=None,
                   help="filter: drop outliers whose similarity to the closest normal prompt (sim_normal_max "
                        "in scores.tsv) is above this, i.e. ones that still look normal. Default: keep all")
    p.add_argument("--min_src_sim", type=float, default=None,
                   help="filter: drop outliers whose image similarity to their source patch (sim_src) is "
                        "below this, i.e. ones that drifted too far to be tissue-like")
    p.add_argument("--min_margin", type=float, default=None,
                   help="optional extra filter: keep outliers with sim_meta_max - sim_normal_max >= this")
    p.add_argument("--preview_grid", default=None,
                   help="dry run only: save a PNG of images just kept / just dropped at --max_normal_sim")
    p.add_argument("--test_feats", default=None, help=".npy, or a raw memmap with --test_dim/--test_dtype")
    p.add_argument("--test_dim", type=int, default=1536)
    p.add_argument("--test_dtype", default="float16")
    p.add_argument("--test_labels", default=None, help="optional .npy of 0/1 (1 = tumor) for a quick AUROC")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--normal_ratio", type=float, default=2.0,
                   help="normals per kept outlier in each epoch (2 = 2:1). 0 = RODEO balancing "
                        "(all normals, outliers resampled to match)")
    p.add_argument("--fresh_normals", action="store_true",
                   help="draw a new random normal subset every epoch instead of one fixed subset")
    p.add_argument("--no_l2norm", action="store_true",
                   help="do not L2-normalise feature rows (default: normalise all sources, like DN2)")
    p.add_argument("--head", choices=["linear", "mlp"], default="linear")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="load and split data, print sizes, write nothing")
    return p.parse_args(argv)


def auroc(labels, scores):
    """Mann-Whitney AUROC with average ranks for ties (no sklearn needed)."""
    labels, scores = np.asarray(labels).astype(bool), np.asarray(scores, dtype=np.float64)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_feats(path, dim=1536, dtype="float16"):
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=dtype, mode="r").reshape(-1, dim)


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def make_head(kind, dim, hidden):
    if kind == "linear":
        return nn.Linear(dim, 2)
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 2))


L2NORM = True  # set from --no_l2norm in main()


def l2n(x):
    """Row-wise L2 normalisation. DN2's GigaPath features are stored L2-normalised; features
    from other extractions may not be. Normalising everything makes all sources comparable
    (a no-op for rows that are already unit length)."""
    return F.normalize(x.float(), dim=-1) if L2NORM else x.float()


@th.no_grad()
def predict(head, X, mu, sd, dev, batch=8192):
    out = []
    for i in tqdm(range(0, len(X), batch), desc="predict", unit="batch", dynamic_ncols=True,
                  mininterval=5, leave=False, disable=len(X) < 50 * batch):
        x = (l2n(th.from_numpy(np.asarray(X[i:i + batch], dtype=np.float32)).to(dev)) - mu) / sd
        out.append(F.softmax(head(x), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def save_threshold_grid(sc, sim_n, thr, path, n=8, size=128):
    """Top row: the n kept images closest to the threshold; bottom row: the n dropped ones."""
    from PIL import Image
    order = np.argsort(sim_n)
    kept = [i for i in order if sim_n[i] <= thr][-n:]
    dropped = [i for i in order if sim_n[i] > thr][:n]
    canvas = Image.new("RGB", (n * size, 2 * size), "white")
    for row, ids in enumerate((kept, dropped)):
        for col, i in enumerate(ids):
            im = Image.open(sc[i]["file"]).convert("RGB").resize((size, size), Image.NEAREST)
            canvas.paste(im, (col * size, row * size))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    canvas.save(path)


def main(argv=None):
    args = parse_args(argv)
    global L2NORM
    L2NORM = not args.no_l2norm
    rng = random.Random(args.seed)
    th.manual_seed(args.seed)
    dev = th.device(args.device)
    if os.path.exists(os.path.join(args.out_dir, "head.pt")) and not args.overwrite and not args.dry_run:
        sys.exit(f"{args.out_dir}/head.pt exists; pass --overwrite to retrain.")

    Xn = load_feats(args.normal_feats)
    Xo = load_feats(args.outlier_feats)
    man, sc = read_tsv(args.normal_manifest), read_tsv(args.outlier_scores)
    assert len(man) == len(Xn), f"manifest rows {len(man)} != normal feats {len(Xn)}"
    assert len(sc) == len(Xo), f"scores rows {len(sc)} != outlier feats {len(Xo)}"

    slide_of = {r["path"]: r["slide"] for r in man}
    slides = sorted(set(slide_of.values()))
    rng.shuffle(slides)
    val_slides = set(slides[: max(1, round(args.val_frac * len(slides)))])

    n_idx = np.arange(len(Xn))
    n_val = np.array([r["slide"] in val_slides for r in man])
    sim_n = np.array([float(r["sim_normal_max"]) for r in sc])
    margin = np.array([float(r["margin"]) for r in sc])
    keep = np.ones(len(sc), dtype=bool)
    if args.max_normal_sim is not None:
        if np.isnan(sim_n).all():
            sys.exit("scores.tsv has no similarity scores (generated with --guide none?)")
        keep &= sim_n <= args.max_normal_sim
    if args.min_margin is not None:
        keep &= margin >= args.min_margin
    if args.min_src_sim is not None:
        keep &= np.array([float(r["sim_src"]) for r in sc]) >= args.min_src_sim
    o_val = np.array([slide_of.get(r["src"], "") in val_slides for r in sc])
    n_ok = np.ones(len(Xn), dtype=bool) if args.normal_ok is None else np.load(args.normal_ok).astype(bool)
    assert len(n_ok) == len(Xn), f"ok rows {len(n_ok)} != normal feats {len(Xn)}"
    tr_n, va_n = n_idx[~n_val & n_ok], n_idx[n_val & n_ok]
    tr_o, va_o = np.where(keep & ~o_val)[0], np.where(keep & o_val)[0]

    print("=" * 70)
    print("train_classifier.py")
    for k, v in sorted(vars(args).items()):
        print(f"  {k:16s} {v}")
    print(f"  slides           {len(slides)} ({len(val_slides)} held out)")
    print(f"  normal           train {len(tr_n)}  val {len(va_n)}")
    print(f"  outliers         train {len(tr_o)}  val {len(va_o)}  (kept {keep.sum()}/{len(sc)})")
    print("=" * 70, flush=True)
    if len(tr_o) == 0:
        sys.exit("no training outliers after filtering/split")
    if args.dry_run:
        if not np.isnan(sim_n).all():
            qs = [0.05, 0.25, 0.5, 0.75, 0.95]
            print("[dry] sim_normal_max quantiles " + "  ".join(f"q{int(q*100)}={np.nanquantile(sim_n, q):.4f}" for q in qs))
            print("[dry] margin quantiles         " + "  ".join(f"q{int(q*100)}={np.nanquantile(margin, q):.4f}" for q in qs))
            for q in (0.9, 0.75, 0.5, 0.25):
                t = np.nanquantile(sim_n, q)
                print(f"[dry] --max_normal_sim {t:.4f} would keep {int((sim_n <= t).sum())}/{len(sim_n)}")
        if args.preview_grid and args.max_normal_sim is not None:
            save_threshold_grid(sc, sim_n, args.max_normal_sim, args.preview_grid)
            print(f"[dry] grid -> {args.preview_grid} (top: just kept, bottom: just dropped)")
        print("[dry] nothing written.")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    # Features stay fp16 on the GPU (1.2M x 1536 normals = ~3.6 GB); each batch is gathered
    # and standardised on the fly, so no full fp32 copy is ever made (that would be ~25 GB).
    def to_gpu_l2(X, idx, chunk=200000):
        parts = []
        for i in range(0, len(idx), chunk):
            x = th.from_numpy(np.ascontiguousarray(X[idx[i:i + chunk]], dtype=np.float32)).to(dev)
            parts.append(l2n(x).half())
        return th.cat(parts)

    tn = to_gpu_l2(Xn, tr_n)
    to = to_gpu_l2(Xo, tr_o)
    print(f"mean row norm before L2: normal {np.linalg.norm(np.asarray(Xn[tr_n[:2000]], np.float32), axis=1).mean():.3f}"
          f"  outlier {np.linalg.norm(np.asarray(Xo[tr_o[:2000]], np.float32), axis=1).mean():.3f}"
          f"  ({'normalised to 1' if L2NORM else 'NOT normalised (--no_l2norm)'})", flush=True)
    s1 = tn.float().sum(0) + to.float().sum(0)
    s2 = tn.float().pow(2).sum(0) + to.float().pow(2).sum(0)
    n_all = len(tn) + len(to)
    mu = s1 / n_all
    sd = (s2 / n_all - mu.pow(2)).clamp_min(0).sqrt() + 1e-6

    head = make_head(args.head, Xn.shape[1], args.hidden).to(dev)
    opt = th.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training set composition
    #   --normal_ratio R : every kept outlier once per epoch + R x as many normals (default 2:1).
    #                      Normals are a fixed random subset unless --fresh_normals, which draws a
    #                      new subset each epoch (same ratio, more normal diversity over training).
    #   --normal_ratio 0 : RODEO's original balancing (all normals, outliers resampled 1:1).
    gen = th.Generator(device="cpu").manual_seed(args.seed)
    if args.normal_ratio > 0:
        n_use = min(len(tn), int(round(args.normal_ratio * len(to))))
        fixed_n = th.randperm(len(tn), generator=gen)[:n_use].to(dev)
        print(f"training set per epoch: {n_use} normals + {len(to)} outliers "
              f"(ratio {n_use / len(to):.2f}:1{', fresh normals each epoch' if args.fresh_normals else ''})",
              flush=True)
    else:
        print(f"training set per epoch: {len(tn)} normals + {len(tn)} outliers resampled 1:1 (RODEO)", flush=True)

    for ep in range(1, args.epochs + 1):
        head.train()
        if args.normal_ratio > 0:
            n_idx_ep = (th.randperm(len(tn), generator=gen)[:n_use].to(dev) if args.fresh_normals else fixed_n)
            o_idx_ep = th.arange(len(to), device=dev)
        else:
            n_idx_ep = th.arange(len(tn), device=dev)
            o_idx_ep = th.randint(0, len(to), (len(tn),), generator=gen).to(dev)
        # one shuffled pool, label = 0 for normal, 1 for outlier; batches keep the pool's ratio
        pool_src = th.cat([th.zeros(len(n_idx_ep), dtype=th.long, device=dev),
                           th.ones(len(o_idx_ep), dtype=th.long, device=dev)])
        pool_idx = th.cat([n_idx_ep, o_idx_ep])
        order = th.randperm(len(pool_idx), generator=gen).to(dev)
        pool_src, pool_idx = pool_src[order], pool_idx[order]
        tot, seen = 0.0, 0
        pbar = tqdm(range(0, len(pool_idx), args.batch), desc=f"epoch {ep}", unit="batch",
                    dynamic_ncols=True, mininterval=5, leave=False)
        for i in pbar:
            ys = pool_src[i:i + args.batch]
            ids = pool_idx[i:i + args.batch]
            xb = th.empty(len(ids), tn.shape[1], dtype=th.float16, device=dev)
            m0 = ys == 0
            xb[m0] = tn[ids[m0]]
            xb[~m0] = to[ids[~m0]]
            xb = (xb.float() - mu) / sd
            loss = F.cross_entropy(head(xb), ys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(ys)
            seen += len(ys)
        head.eval()
        val = float("nan")
        if len(va_n) and len(va_o):
            pv = np.concatenate([predict(head, Xn[va_n], mu, sd, dev), predict(head, Xo[va_o], mu, sd, dev)])
            val = auroc(np.r_[np.zeros(len(va_n)), np.ones(len(va_o))], pv)
        print(f"epoch {ep:3d}  loss {tot / seen:.4f}  val AUROC (normal vs synthetic) {val:.4f}", flush=True)

    th.save({"head": head.state_dict(), "mu": mu.cpu(), "sd": sd.cpu(), "args": vars(args)},
            os.path.join(args.out_dir, "head.pt"))

    test_auc = float("nan")
    if args.test_feats:
        Xt = load_feats(args.test_feats, args.test_dim, args.test_dtype)
        st = predict(head, Xt, mu, sd, dev)
        np.save(os.path.join(args.out_dir, "test_scores.npy"), st)
        print(f"test scores -> {args.out_dir}/test_scores.npy ({len(st)} rows)")
        if args.test_labels:
            test_auc = auroc(np.load(args.test_labels), st)
            print(f"test patch AUROC {test_auc:.4f}")

    with open(os.path.join(args.out_dir, "summary.tsv"), "w") as f:
        f.write("key\tvalue\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}\t{v}\n")
        f.write(f"n_train_normal_pool\t{len(tr_n)}\nn_train_normal_used\t"
                f"{n_use if args.normal_ratio > 0 else len(tr_n)}\nn_train_outlier\t{len(tr_o)}\n"
                f"final_val_auroc\t{val:.4f}\ntest_patch_auroc\t{test_auc:.4f}\n")


if __name__ == "__main__":
    main()