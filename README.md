# RODEO for metastasis detection in histopathology

This repo adapts RODEO (Mirzaei et al., ICML 2024) to lymph node metastasis detection on
CAMELYON16. RODEO is an outlier exposure method: it trains a classifier on normal data plus
synthetic "near-outliers" made by a text-guided diffusion model. I wanted to see whether the
idea carries over to histopathology, where you can't simply ask a generic image model for
"tumour".

Short version: it doesn't, at least not with current text-conditioned generators. A simple
kNN baseline on the same features did clearly better. Most of the work here ended up being
about understanding why.

<p align="center">
  <img src="assets/generated_<img width="1280" height="569" alt="photo_2026-10-24 18 51 44" src="https://github.com/user-attachments/assets/d9b79059-69a5-4c95-9fc3-af7bc048f83b" />
grid.png" width="650"><br>
  <em>Top: real normal patches. Bottom: the synthetic outliers generated from them.</em>
</p>

## What the pipeline does

Normal patches are partially noised and then denoised by a diffusion model conditioned on
metastasis descriptions from `pathology_terms.txt` (for example "discohesive cells" or
"glandular arrangement"). I tried two generators: stock GLIDE with its noise-aware CLIP
guidance, which is exactly RODEO's setup, and PathLDM, a latent diffusion model trained on
pathology and conditioned on PLIP. A linear head on frozen Prov-GigaPath features then
learns to separate real normals from the generated images. Its output is the anomaly score
on real test patches.

The main differences from the original RODEO are frozen foundation-model features instead
of a ResNet trained from scratch, no adversarial training, and an optional CLIP/PLIP
similarity filter on the generated images.

## Results

Patch-level AUROC, trained on normal patches only.

| Method | CAMELYON16 | BMAD CAMELYON16 |
|---|---|---|
| kNN on GigaPath features (baseline) | 95.2 | 84.6 |
| RODEO + GLIDE | 73.1 | 68.3 |
| RODEO + PathLDM | – | 68.8 |
| Same head trained on 119 real tumour patches | – | 89.3 |

Filtering the generated images with CLIP or PLIP similarity didn't help, and a
pathology-specific generator did no better than GLIDE.

The last row is the useful one. When the head sees a handful of real tumour patches, it
beats kNN, so the classifier isn't the problem; the synthetic outliers are. A few
controlled experiments on BMAD narrowed it down:

- **Artifacts.** If I use images that only went through PathLDM's encoder and decoder, with
  no editing at all, as the outliers, I already get about 70 AUROC. That's roughly what
  every RODEO variant reached. The classifier is mostly learning "this image came out of a
  generator".
- **Too much editing.** At the default strength, the edited images end up much farther from
  the normals than real tumours are. They keep so little of their source patch that two
  random normal patches resemble each other more. Lower strengths (around 0.3) land at
  about the right distance.
- **Prompts.** When I make the artifacts and the editing process identical in both classes,
  so that only the tumour prompt differs, performance falls to chance. The prompts change
  the images consistently, just not toward what real lymph node metastases look like.

My takeaway is that synthetic outliers in pathology need to share the normal class's
artifacts and move toward real tumour morphology, and text prompts alone don't get you the
second part.

## Files

| File | Purpose |
|---|---|
| `extract_patches.py` | Tissue patches from normal training slides |
| `generate_outliers.py` | GLIDE outlier generation (CLIP, PLIP, or no guidance) |
| `score_plip.py` | PLIP similarity scores for filtering generated images |
| `extract_gigapath.py` | Prov-GigaPath features |
| `train_classifier.py` | Normal-vs-outlier head and test scoring |
| `train_glide.py`, `caption_patches.py` | Optional GLIDE fine-tuning on H&E |
| `glide_common.py` | Shared helpers |
| `pathology_terms.txt` | Metastasis and normal tissue vocabulary with prompt templates |
| `bmad/` | PathLDM generation and the diagnostic experiments |

Each script has `--help` and `--dry_run`, and can be rerun to resume after a crash.

## Running it

```bash
pip install -r requirements.txt

python extract_patches.py --slide_glob "data/camelyon16/training/normal/*.tif" --out_dir data/c16_normal

python generate_outliers.py --base_ckpt openai --no_upsample --guide clip --guide_scale 10 \
    --t_min 0.5 --t_max 0.5 --manifest data/c16_normal/manifest.tsv \
    --prompts_file pathology_terms.txt --out_dir gen/rodeo --n_images 20000

python extract_gigapath.py --list data/c16_normal/manifest.tsv --col path --out feats/normal --degrade 64
python extract_gigapath.py --list gen/rodeo/scores.tsv --col file --out feats/gen

python train_classifier.py --normal_feats feats/normal.npy --normal_manifest data/c16_normal/manifest.tsv \
    --outlier_feats feats/gen.npy --outlier_scores gen/rodeo/scores.tsv \
    --test_feats <test_feats.npy> --test_labels <labels.npy> --out_dir runs/cls
```

CAMELYON16 is available from the [challenge site](https://camelyon16.grand-challenge.org/).
Prov-GigaPath needs approved access on Hugging Face.

## Limitations

Each setting was run once, so there are no error bars. I didn't use RODEO's adversarial
training. GLIDE works at 64px, so real patches were downsampled to match. PLIP as the
guidance model for GLIDE is implemented but I haven't evaluated it yet.

## References

RODEO (Mirzaei et al., ICML 2024), GLIDE (Nichol et al., ICML 2022), PathLDM
(Yellapragada et al., WACV 2024), PLIP (Huang et al., Nature Medicine 2023),
Prov-GigaPath (Xu et al., Nature 2024), BMAD (Bao et al., CVPRW 2024),
CAMELYON16 (Ehteshami Bejnordi et al., JAMA 2017), DN2 (Bergman et al., 2020).

## License

MIT for the code here. Models and datasets keep their own licenses.
