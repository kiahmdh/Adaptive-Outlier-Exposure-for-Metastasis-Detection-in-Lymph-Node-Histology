"""Shared helpers for the RODEO-on-Camelyon GLIDE pipeline.

Imported by train_glide.py and generate_outliers.py. Not meant to be run directly.
"""
import ast
import random

import numpy as np

# GLIDE's CLIP code uses the alias np.bool, which numpy 1.24 removed (numpy 2 re-added it).
# Plain bool is exactly what it meant; set it before glide_text2im.clip is imported.
if "bool" not in np.__dict__:
    np.bool = bool

import torch as th
import torch.nn.functional as F
from PIL import Image

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)

import glide_text2im.unet as _glide_unet


def _upsample_forward(self, x):
    """GLIDE's Upsample with the nearest-neighbour step done in fp32.

    torch 2.0.1 has no bf16 CUDA kernel for upsample_nearest2d, which breaks bf16
    autocast. Doing the (exact, weightless) upsampling in fp32 and casting back is
    numerically identical; the following conv still runs under autocast.
    """
    assert x.shape[1] == self.channels
    dtype = x.dtype
    x = x.float()
    if self.dims == 3:
        x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
    else:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
    x = x.to(dtype)
    if self.use_conv:
        x = self.conv(x)
    return x


_glide_unet.Upsample.forward = _upsample_forward
th.backends.cuda.matmul.allow_tf32 = True  # faster fp32 paths on the 4090, no effect on bf16
th.backends.cudnn.allow_tf32 = True

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# --------------------------------------------------------------------------- prompts
def load_prompt_lists(path):
    """Parse pathology_terms.txt (Python list assignments) without exec."""
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    for key in ("metastasis_text_prompts", "normal_text_prompts", "templates"):
        if key not in out:
            raise KeyError(f"{key} not found in {path}")
    # the file has exact duplicates (e.g. "microacinar structures" x3); keep first occurrence
    for key in ("metastasis_text_prompts", "normal_text_prompts"):
        seen, uniq = set(), []
        for t in out[key]:
            k = t.strip().lower()
            if k not in seen:
                seen.add(k)
                uniq.append(t.strip())
        out[key] = uniq
    return out


def fill(template, term):
    return template.replace("CLASSNAME", term)


# --------------------------------------------------------------------------- GLIDE
def build_glide(stage, timestep_respacing=""):
    opts = model_and_diffusion_defaults_upsampler() if stage == "upsample" else model_and_diffusion_defaults()
    opts["use_fp16"] = False  # we use torch autocast instead of GLIDE's fp16 path
    opts["timestep_respacing"] = timestep_respacing
    model, diffusion = create_model_and_diffusion(**opts)
    return model, diffusion, opts


def load_openai_weights(model, stage, cache_dir):
    name = "upsample" if stage == "upsample" else "base"
    model.load_state_dict(load_checkpoint(name, th.device("cpu"), cache_dir=cache_dir))


def respaced_diffusion(opts, respacing):
    return create_gaussian_diffusion(
        steps=opts["diffusion_steps"],
        noise_schedule=opts["noise_schedule"],
        timestep_respacing=respacing,
    )


def tokenize(model, captions, text_ctx, device):
    toks, masks = [], []
    for c in captions:
        t = model.tokenizer.encode(c) if c else []
        t, m = model.tokenizer.padded_tokens_and_mask(t, text_ctx)
        toks.append(t)
        masks.append(m)
    return th.tensor(toks, device=device), th.tensor(masks, dtype=th.bool, device=device)


def make_model_fn(model, cond_tokens, cond_mask, uncond_tokens=None, uncond_mask=None,
                  cfg_scale=1.0, low_res=None, amp_dtype=None):
    """Returns f(x, t) -> [eps, var_values] with optional classifier-free guidance.

    Works with SpacedDiffusion.p_mean_variance (which remaps t before calling us).
    """
    use_cfg = cfg_scale is not None and cfg_scale > 1.0 and uncond_tokens is not None

    def fn(x, t, **_):
        extra = {} if low_res is None else {"low_res": low_res}
        with th.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None and x.is_cuda):
            if not use_cfg:
                out = model(x, t, tokens=cond_tokens, mask=cond_mask, **extra)
            else:
                if low_res is not None:
                    extra = {"low_res": th.cat([low_res, low_res], 0)}
                out2 = model(
                    th.cat([x, x], 0), th.cat([t, t], 0),
                    tokens=th.cat([cond_tokens, uncond_tokens], 0),
                    mask=th.cat([cond_mask, uncond_mask], 0), **extra,
                )
                cond, unc = out2.chunk(2, dim=0)
                eps = unc[:, :3] + cfg_scale * (cond[:, :3] - unc[:, :3])
                out = th.cat([eps, cond[:, 3:]], dim=1)
        return out.float()

    return fn


def sample_loop(diffusion, model_fn, x_init, t_start, guide=None, guide_scale=0.0):
    """Ancestral sampling in respaced index space, from step t_start-1 down to 0.

    x_init must already be at noise level t_start-1 (pure noise if t_start == num_timesteps).
    guide: a GlideClipGuide / PlipGuide (see below) returning a per-sample objective.
    The guidance is applied exactly like GLIDE's / RODEO's cond_fn:
        mean <- mean + variance * guide_scale * d(objective)/d(x_t)
    """
    x = x_init
    tmap = diffusion.timestep_map
    for i in reversed(range(t_start)):
        t = th.full((x.shape[0],), i, device=x.device, dtype=th.long)
        if guide is not None and guide_scale > 0:
            with th.enable_grad():
                xg = x.detach().requires_grad_(True)
                if guide.needs_x0:  # PLIP: score the model's clean-image prediction
                    out = diffusion.p_mean_variance(model_fn, xg, t, clip_denoised=True)
                    obj = guide(xg, tmap[i], out["pred_xstart"])
                else:               # GLIDE noised CLIP: score x_t directly at its timestep
                    with th.no_grad():
                        out = diffusion.p_mean_variance(model_fn, x, t, clip_denoised=True)
                    obj = guide(xg, tmap[i], None)
                g = th.autograd.grad(obj.sum(), xg)[0]
            mean = out["mean"].detach().float() + out["variance"].detach().float() * guide_scale * g.float()
            log_var = out["log_variance"].detach()
        else:
            with th.no_grad():
                out = diffusion.p_mean_variance(model_fn, x, t, clip_denoised=True)
            mean, log_var = out["mean"], out["log_variance"]
        if i > 0:
            x = mean + th.exp(0.5 * log_var) * th.randn_like(x)
        else:
            x = mean
    return x.detach().clamp(-1, 1)


# --------------------------------------------------------------------------- guidance encoders
class GlideClipGuide:
    """What RODEO uses: GLIDE's noise-aware CLIP (clip/image-enc + clip/text-enc), 64x64 input.

    Objective = exp(logit_scale) * <z_image(x_t, t), z_text>, identical to GLIDE's cond_fn.
    """
    needs_x0 = False
    name = "clip"

    def __init__(self, device, cache_dir=None):
        from glide_text2im.clip.model_creation import create_clip_model
        self.device = device
        self.clip = create_clip_model(device=device)
        self.clip.image_encoder.load_state_dict(load_checkpoint("clip/image-enc", device, cache_dir=cache_dir))
        self.clip.text_encoder.load_state_dict(load_checkpoint("clip/text-enc", device, cache_dir=cache_dir))
        for m in (self.clip.image_encoder, self.clip.text_encoder):
            m.eval().requires_grad_(False)
        self.z_t = None

    @th.no_grad()
    def embed_texts(self, texts, batch=256):
        return th.cat([self.clip.text_embeddings(texts[i:i + batch]) for i in range(0, len(texts), batch)]).float()

    def set_prompts(self, captions):
        self.z_t = self.embed_texts(captions)

    def __call__(self, x_t, t_orig, pred_x0):
        t = th.full((x_t.shape[0],), int(t_orig), device=x_t.device, dtype=th.long)
        z_i = self.clip.image_embeddings(x_t, t)
        return th.exp(self.clip.logit_scale) * (z_i * self.z_t).sum(-1)

    @th.no_grad()
    def embed_images(self, x_m11):
        """Clean images (t=0), any size; resized to the encoder's 64x64."""
        if x_m11.shape[-1] != 64:
            x_m11 = F.interpolate(x_m11.float(), size=(64, 64), mode="area")
        t = th.zeros(x_m11.shape[0], device=x_m11.device, dtype=th.long)
        return self.clip.image_embeddings(x_m11, t).float()


class PlipGuide:
    """The swap: PLIP. Not noise-aware, so it scores the model's predicted clean image."""
    needs_x0 = True
    name = "plip"

    def __init__(self, device, name="vinid/plip", n_aug=1):
        self.plip = PLIP(device, name)
        self.n_aug = n_aug
        self.scale = self.plip.model.logit_scale.exp().float().item()
        self.z_t = None

    def embed_texts(self, texts):
        return self.plip.text(texts)

    def set_prompts(self, captions):
        self.z_t = self.embed_texts(captions)

    def __call__(self, x_t, t_orig, pred_x0):
        s = 0.0
        for k in range(self.n_aug):  # dihedral views; n_aug=1 means no augmentation
            v = th.rot90(pred_x0, k % 4, dims=(2, 3))
            v = th.flip(v, dims=(3,)) if k >= 4 else v
            s = s + (self.plip.image(v) * self.z_t).sum(-1)
        return self.scale * s / self.n_aug

    @th.no_grad()
    def embed_images(self, x_m11):
        return self.plip.image(x_m11)


def term_ensembles(encoder, terms, templates):
    """One L2-normalised text embedding per term, averaged over all templates."""
    texts = [fill(tp, t) for t in terms for tp in templates]
    e = encoder.embed_texts(texts).view(len(terms), len(templates), -1).mean(1)
    return F.normalize(e, dim=-1)


# --------------------------------------------------------------------------- PLIP
class PLIP:
    """HuggingFace PLIP (vinid/plip). Image path is differentiable for guidance."""

    def __init__(self, device, name="vinid/plip"):
        from transformers import AutoTokenizer, CLIPModel
        self.device = device
        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # vinid/plip ships only tokenizer.json, so the slow CLIPTokenizer fails; the fast one works
        self.tok = AutoTokenizer.from_pretrained(name)
        self.mean = th.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = th.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    @th.no_grad()
    def text(self, texts, batch=256):
        feats = []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=77,
                           return_tensors="pt").to(self.device)
            feats.append(self.model.get_text_features(**enc))
        return F.normalize(th.cat(feats, 0).float(), dim=-1)

    @th.no_grad()
    def term_ensembles(self, terms, templates):
        """One L2-normalised embedding per term, averaged over all templates."""
        texts = [fill(tp, t) for t in terms for tp in templates]
        e = self.text(texts).view(len(terms), len(templates), -1).mean(1)
        return F.normalize(e, dim=-1)

    def image(self, x_m11):
        """x in [-1, 1], any size, (B,3,H,W). Differentiable."""
        x = (x_m11.float() + 1) / 2
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        return F.normalize(self.model.get_image_features(pixel_values=x).float(), dim=-1)


# --------------------------------------------------------------------------- image io
def pil_to_m11(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return th.from_numpy(a).permute(2, 0, 1)


def m11_to_uint8(x):
    x = ((x.detach().float().clamp(-1, 1) + 1) * 127.5).round().to(th.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def quantize_m11(x):
    """What GLIDE does to the base output before feeding it to the upsampler."""
    return ((x + 1) * 127.5).round() / 127.5 - 1


def save_grid(rows, path, size=128):
    """rows: list of (B,3,H,W) tensors in [-1,1]; each row resized to `size`."""
    tiles = []
    for r in rows:
        r = F.interpolate(r.float(), size=(size, size), mode="nearest")
        tiles.append(np.concatenate(list(m11_to_uint8(r)), axis=1))
    Image.fromarray(np.concatenate(tiles, axis=0)).save(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    th.manual_seed(seed)