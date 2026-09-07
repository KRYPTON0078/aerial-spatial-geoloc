# Prepare University-1652 (checklist)

Do this on Magne’s machine after academic access is granted. CI and clones of this repo **cannot** fetch the imagery.

## 1. Request access (pick one; academic email)

- [ ] Open [Request.md](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).
- [ ] Email **zdzheng12@gmail.com** from your university address.
- [ ] Subject: `Request of University1652 Dataset`
- [ ] Body: research-only, no redistribution, no commercial use (as in Request.md).
- [ ] **Or** request access on Hugging Face: [layumi/university-1652](https://huggingface.co/datasets/layumi/university-1652) (gated; agree to research-only terms).

Do **not** ask this repo to “just download it.” Without approval there is no legal copy.

## 2. Extract locally (do not commit)

Expected size is **multi-GB**. Keep it outside git.

- [ ] Extract so you have `train/drone/` and `test/gallery_satellite/`.
- [ ] Typical names: `University-1652/` or `University-Release/`.
- [ ] Suggested path (already gitignored): `data/university1652/`
- [ ] Confirm the Baseline layout in [`docs/DATASETS.md`](../docs/DATASETS.md).

Hugging Face, **only after** access is granted:

```bash
huggingface-cli login
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='layumi/university-1652', repo_type='dataset', local_dir='data/university1652', local_dir_use_symlinks=False)"
```

## 3. Point the config

- [ ] Set `data.root` in `configs/default.yaml` to that folder (or the nested `University-1652` directory inside it). The adapter also probes `University-Release/` and `university-1652/`.
- [ ] `dataset: university1652` (already the default config).

## 4. Train / eval here (then write numbers)

- [ ] `python -m aerial_geoloc.cli train --config configs/default.yaml`
- [ ] `python -m aerial_geoloc.cli eval --config configs/default.yaml --checkpoint outputs/university1652_resnet18/best.pt`
- [ ] **TODO:** copy *those* measured R@1 / mAP into README / `reports/` — not literature tables.

Until step 4 is done, published figures in this repo are **simulation-only**.
