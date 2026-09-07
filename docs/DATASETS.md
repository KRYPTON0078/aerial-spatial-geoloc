# Datasets

## University-1652 (gated — do not fake a download)

**Zheng, Wei, Yang. University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization. ACM MM 2020.** [doi](https://doi.org/10.1145/3394171.3413896)

Academic, **non-commercial** research only. Imagery originates from Google Maps/Earth. Cite the paper in any work that uses the data, models trained on it, or numbers measured on it.

This repository **does not** contain University-1652 pixels, **does not** download them, and **does not** report University-1652 Recall@K / mAP. Those metrics stay **TODO** until Magne obtains the release and runs `configs/default.yaml` locally.

### How to obtain access

Two official routes (same authors). Use your **academic email**.

1. **Email (Request.md)** — copy the template at
   [Request.md](https://github.com/layumi/University1652-Baseline/blob/master/Request.md)
   and send it to **zdzheng12@gmail.com** with title `Request of University1652 Dataset`.
   The authors typically reply with a download address.

2. **Hugging Face (gated)** — after you are approved, download the raw folders (not `load_dataset()` parquet):
   [huggingface.co/datasets/layumi/university-1652](https://huggingface.co/datasets/layumi/university-1652)

   ```bash
   # only after HF access is granted on your account
   huggingface-cli login
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='layumi/university-1652', repo_type='dataset', local_dir='data/university1652', local_dir_use_symlinks=False)"
   ```

   If the snapshot wraps an extra `University-1652/` directory, point `data.root` at that inner folder (the adapter also searches common nested names).

Full checklist: [`scripts/prepare_u1652.md`](../scripts/prepare_u1652.md).

### Exact directory layout (University1652-Baseline)

Point `data.root` in `configs/default.yaml` at the folder that contains `train/` and `test/` (often named `University-1652` or `University-Release`):

```
<data.root>/
  train/
    drone/<building_id>/*.jpg
    satellite/<building_id>/*.jpg
    street/<building_id>/*.jpg
    google/<building_id>/*.jpg      # optional noisy web street (Baseline `--extra`)
  test/
    query_drone/<building_id>/*.jpg
    query_satellite/<building_id>/*.jpg
    query_street/<building_id>/*.jpg
    gallery_drone/<building_id>/*.jpg
    gallery_satellite/<building_id>/*.jpg
    gallery_street/<building_id>/*.jpg
    4K_drone/                       # optional; ignored by default splits
```

Required for this adapter: `train/drone`, `train/satellite`, `test/query_drone`, `test/gallery_satellite`.
Street / google / extra test views are loaded when present.

Never git-add that tree. `.gitignore` already excludes `data/university1652/`, `data/University-Release/`, and `data/University-1652/`.

### After you have the files

```bash
python -m aerial_geoloc.cli train --config configs/default.yaml
python -m aerial_geoloc.cli eval --config configs/default.yaml --checkpoint outputs/university1652_resnet18/best.pt
```

Standard tasks (Baseline):

- **Drone → Satellite** (drone-view target localization): `query_view: drone`, `gallery_view: satellite` (default YAML).
- **Satellite → Drone** (drone navigation): swap those two fields.

**TODO (not done in this repo):** fill README / `reports/` with measured University-1652 R@1 and mAP from *this* training run. Do not copy LPN / FSRA / WeatherPrompt tables.

## Synthetic campus (shipped)

`data/sim/` is the interim study split (pose-labeled, ~1.5 MB). Reproduce with `python -m aerial_geoloc.cli study --config configs/study.yaml`. It uses the same *folder contract* as U1652 so the adapter path can be tested without the gated release. **Not a substitute for University-1652 numbers.**

`python -m aerial_geoloc.cli make-demo-data` writes a still-smaller geometric demo.

## SUES-200

Documented stub only: `aerial_geoloc.data.sues200`. Not loaded by the CLI yet.
