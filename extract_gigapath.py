#!/usr/bin/env python
"""Phase 3 input: Prov-GigaPath tile features for a list of images.

Official GigaPath tile preprocessing: resize 256 (bicubic) -> center crop 224 ->
ImageNet normalisation. Any input size works (64px outputs are upscaled first).

--list is a TSV/CSV with a column of image paths (--col): e.g. the patch
manifest.tsv (col "path") or a generation scores.tsv (col "file").

Output: <out>.npy (N x 1536, float16, same row order as --list) and <out>.paths.txt
Resumable in chunks (<out>_parts/); rerun the same command after a crash.
"""
import argparse
import csv
import glob
import io
import os
import shutil

import numpy as np
import torch as th
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", required=True)
    p.add_argument("--col", default="path")
    p.add_argument("--out", required=True, help="output prefix, e.g. feats/normal_train")
    p.add_argument("--degrade", type=int, default=0,
                   help="0 = off. N = put every image through the same path as the generated outliers "
                        "before encoding: box-downsample to NxN, JPEG round trip (--degrade_quality), then "
                        "the usual resize-256 / crop-224. Use 64 to match --no_upsample generations.")
    p.add_argument("--degrade_quality", type=int, default=90)
    p.add_argument("--no_l2norm", action="store_true",
                   help="store raw features (default: L2-normalised rows, identical to DN2's extract_features.py)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--chunk", type=int, default=20000, help="rows per resumable part")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="embed one batch, print shape, write nothing")
    return p.parse_args(argv)


class Images(Dataset):
    def __init__(self, paths, degrade=0, quality=90):
        from torchvision import transforms
        self.paths = paths
        self.degrade, self.quality = degrade, quality
        self.tf = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        if self.degrade and im.size != (self.degrade, self.degrade):
            # same as generate_outliers.py: 256 -> NxN (BOX), saved as JPEG
            im = im.resize((256, 256), Image.LANCZOS) if im.size != (256, 256) else im
            im = im.resize((self.degrade, self.degrade), Image.BOX)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=self.quality)
            buf.seek(0)
            im = Image.open(buf).convert("RGB")
        return self.tf(im)


def main(argv=None):
    args = parse_args(argv)
    delim = "\t" if args.list.endswith(".tsv") else ","
    with open(args.list) as f:
        paths = [r[args.col] for r in csv.DictReader(f, delimiter=delim)]
    parts_dir = args.out + "_parts"
    final = args.out + ".npy"

    print("=" * 70)
    print("extract_gigapath.py")
    for k, v in sorted(vars(args).items()):
        print(f"  {k:10s} {v}")
    print(f"  {'n_images':10s} {len(paths)}")
    print("=" * 70, flush=True)
    if os.path.exists(final) and not args.overwrite and not args.dry_run:
        print(f"{final} exists; pass --overwrite to redo.")
        return
    if args.overwrite and os.path.isdir(parts_dir):
        shutil.rmtree(parts_dir)

    import timm
    dev = th.device(args.device)
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(dev).eval()

    def embed(sub):
        loader = DataLoader(Images(sub, args.degrade, args.degrade_quality), batch_size=args.batch, num_workers=args.workers, pin_memory=True)
        out = []
        with th.no_grad(), th.autocast("cuda", dtype=th.float16, enabled=dev.type == "cuda"):
            for x in loader:
                z = model(x.to(dev, non_blocking=True)).float()
                if not args.no_l2norm:
                    z = th.nn.functional.normalize(z, dim=-1)
                out.append(z.cpu().numpy().astype(np.float16))
                if args.dry_run:
                    break
        return np.concatenate(out)

    if args.dry_run:
        f = embed(paths[: args.batch])
        print(f"[dry] features {f.shape} {f.dtype}, mean |x| {np.abs(f.astype(np.float32)).mean():.4f}")
        return

    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(final)), exist_ok=True)
    n_parts = (len(paths) + args.chunk - 1) // args.chunk
    for k in range(n_parts):
        part = os.path.join(parts_dir, f"part_{k:04d}.npy")
        if os.path.exists(part):
            continue
        f = embed(paths[k * args.chunk:(k + 1) * args.chunk])
        np.save(part + ".tmp.npy", f)
        os.replace(part + ".tmp.npy", part)
        print(f"part {k + 1}/{n_parts} done ({min((k + 1) * args.chunk, len(paths))}/{len(paths)})", flush=True)

    feats = np.concatenate([np.load(p) for p in sorted(glob.glob(os.path.join(parts_dir, "part_*.npy")))])
    assert len(feats) == len(paths), (len(feats), len(paths))
    np.save(final, feats)
    with open(args.out + ".paths.txt", "w") as f:
        f.writelines(p + "\n" for p in paths)
    shutil.rmtree(parts_dir)
    print(f"saved {final} {feats.shape}")


if __name__ == "__main__":
    main()