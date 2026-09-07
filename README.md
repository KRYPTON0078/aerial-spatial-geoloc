# Aerial Spatial Geoloc

**PyTorch toolkit for drone↔satellite cross-view geo-localization, with a CPU simulation study of weather robustness and uncertainty-aware retrieval.**

## Abstract

This repository implements a research baseline for **drone → satellite** instance retrieval: dual-view encoders, InfoNCE + identity loss, a University-1652 folder adapter, and a **pose-labeled simulated campus** used to measure labeled nuisances (fog, rain, night, snow, occlusion, yaw, altitude).

The simulation study compares **clean training** vs **weather-augmented training** and an optional **MC-Dropout uncertainty-aware (UA)** ranker. All figures and tables below are generated from that split (seed 42, TinyCNN, CPU). They are **not** University-1652 scores. The real U1652 protocol is supported by the adapter once a gated extract is present locally; those metrics are left unreported until they are measured here.

Full write-up: [`reports/EXPERIMENT_REPORT.md`](reports/EXPERIMENT_REPORT.md) · metrics: [`reports/metrics.json`](reports/metrics.json) · related work: [`docs/literature_review.md`](docs/literature_review.md)

## 中文摘要

面向无人机–卫星跨视角地理定位的 PyTorch 工具包：双塔 / 共享编码器、InfoNCE 与身份损失、University-1652 目录适配器，以及带位姿与天气标签的**可控仿真校园**。仿真实验对比干净训练与天气增强，并可选 MC-Dropout 不确定性加权检索。干净训练在本仿真测试上 R@1 = 0.719，雾 / 夜降至 0.125；天气增强以干净集精度为代价将夜视 R@1 提升至 0.281。图与表均由仓库内脚本生成。**所列数字不是 University-1652 评测结果。**

## Features

- Dual-view encoder (shared or view-specific backbone) with L2-normalized embeddings
- Losses: symmetric InfoNCE, location-ID classification, optional batch-hard triplet (`configs/ablation_*.yaml`)
- Backbones: TinyCNN (CPU study), ResNet, Tiny ViT
- University-1652 adapter matching [University1652-Baseline](https://github.com/layumi/University1652-Baseline) `train/` / `test/` folders (no imagery shipped)
- Procedural campus (`data/sim/`) with recorded altitude / yaw and the same folder contract
- Named eval nuisances plus train-time weather overlays
- MC-Dropout on the **query** only; gallery embeddings stay deterministic; UA score \(\cos/(1+\sigma)\)
- CLI: `train`, `eval`, `retrieve`, `sim-generate`, `study`; optional Gradio UI extra

## Method

Shared TinyCNN encodes drone and satellite tiles. Training uses in-batch InfoNCE (\(\tau=0.07\)) plus identity CE. Test nuisances are **named**, not unstructured noise. UA scoring draws \(T=6\) dropout samples on the query.

<p align="center">
  <img src="figures/architecture.png" alt="Dual-view encoder with InfoNCE and MC-Dropout uncertainty-aware scoring" width="900"/>
</p>

*Figure 1. Dual-view retrieval: query/gallery encoders, InfoNCE + ID loss, cosine ranking, and MC-Dropout uncertainty-aware scoring.*

## Results

**Synthetic campus only** — 12 train / 8 test locations, 32 drone queries, seed 42, CPU TinyCNN. **Not University-1652.** Source: [`reports/metrics.json`](reports/metrics.json).

| Condition | Baseline R@1 | Weather-aug R@1 | Weather-aug + UA R@1 |
| --- | ---: | ---: | ---: |
| clean | **0.719** | 0.594 | 0.531 |
| fog | 0.125 | 0.156 | **0.188** |
| night | 0.125 | **0.281** | 0.188 |
| occlusion | **0.688** | 0.469 | 0.625 |

Weather augmentation **lowers clean R@1** and **raises night / fog**. UA scoring helps occlusion for the weather-aug model and is mixed elsewhere.

### Figure gallery

PNGs under [`figures/`](figures/) are produced by `python -m aerial_geoloc.cli study` from the committed metrics.

<p align="center">
  <img src="figures/training_curves.png" alt="Train loss and clean-test Recall@1 over six epochs" width="900"/>
</p>

*Figure 2. Training dynamics. Clean training reaches R@1 0.719 then overfits slightly. Weather-aug has a harder loss and a lower clean-test ceiling.*

<p align="center">
  <img src="figures/recall_by_condition.png" alt="Recall@1 bars across eight labeled nuisances" width="900"/>
</p>

*Figure 3. Drone→sat Recall@1 under labeled nuisances. Baseline leads on clean / rain / snow / geometry; weather-aug + UA helps fog; weather-aug cosine helps night.*

<p align="center">
  <img src="figures/map_by_condition.png" alt="mAP bars across eight labeled nuisances" width="900"/>
</p>

*Figure 4. Same protocol, mAP. Night and fog are the conditions where weather-aug beats the clean specialist on this renderer.*

<p align="center">
  <img src="figures/retrieval_grid_fog.png" alt="Fog drone queries with top-5 satellite matches, green correct red wrong" width="900"/>
</p>

*Figure 5. Qualitative retrieval under fog (weather-aug model). Left: query. Right: top-5 satellite tiles. Green border = correct location ID.*

<p align="center">
  <img src="figures/failures_night.png" alt="Night-time failure cases: query, predicted satellite, true satellite" width="720"/>
</p>

*Figure 6. Failures under night (clean-trained baseline): query | predicted satellite | true satellite.*

<p align="center">
  <img src="figures/embedding_pca.png" alt="PCA of drone and satellite embeddings colored by location" width="640"/>
</p>

*Figure 7. PCA of embeddings (circles = drone, stars = satellite), colored by location. Overlapping markers are what dual-view InfoNCE is optimizing for.*

<p align="center">
  <img src="figures/uncertainty_vs_recall.png" alt="Mean MC-Dropout uncertainty versus Recall@1 by condition" width="900"/>
</p>

*Figure 8. Mean MC-Dropout σ (bars) vs R@1 (line) for the weather-aug model. Night/fog remain high-error; σ is not a perfectly calibrated error detector on this encoder.*

## Installation

Python 3.10+ and a CPU-capable PyTorch install.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,study]"
pytest
```

Optional UI: `pip install -e ".[ui]"`.

## Reproduce the study

Shipped copies of `data/sim/`, `figures/*.png`, and `reports/metrics.json` are already in the tree. To regenerate (CPU, minutes):

```bash
python -m aerial_geoloc.cli study --config configs/study.yaml --regenerate
```

Smaller smoke path:

```bash
python -m aerial_geoloc.cli sim-generate --root data/sim
python -m aerial_geoloc.cli train --config configs/demo.yaml
python -m aerial_geoloc.cli retrieve --config configs/demo.yaml --checkpoint outputs/demo/best.pt
```

University-1652 training (after a local extract): `python -m aerial_geoloc.cli train --config configs/default.yaml`.

## Repository layout

```
src/aerial_geoloc/        config, adapters, DualViewEncoder, train / eval
src/aerial_geoloc/sim     pose-labeled campus + nuisances
src/aerial_geoloc/study   CPU experiment runner + figure generation
configs/                  default, demo, study, ablation YAMLs
data/sim/                 shipped simulated split (~1.5 MB, 168 tiles)
figures/                  committed study figures (Fig. 1–8)
reports/                  EXPERIMENT_REPORT.md, metrics.json
docs/                     DATASETS.md, literature_review.md
scripts/prepare_u1652.md  gated-dataset checklist
```

## Dataset: University-1652

University-1652 ([Zheng, Wei, Yang, ACM MM 2020](https://doi.org/10.1145/3394171.3413896)) is **gated**, academic / non-commercial, and is **not** redistributed here.

1. Request via [Request.md](https://github.com/layumi/University1652-Baseline/blob/master/Request.md) to **zdzheng12@gmail.com**, or accept the terms on [Hugging Face `layumi/university-1652`](https://huggingface.co/datasets/layumi/university-1652).
2. Extract so `train/drone` and `test/gallery_satellite` exist; point `data.root` in `configs/default.yaml` at that tree.
3. Checklist and exact Baseline folders: [`scripts/prepare_u1652.md`](scripts/prepare_u1652.md), [`docs/DATASETS.md`](docs/DATASETS.md).

Do not copy LPN / FSRA / WeatherPrompt tables into this README. U1652 R@1 / mAP from *this* code remain **unreported** until a local run exists.

## Citation

If you use University-1652, cite Zheng et al., ACM MM 2020. Weather and uncertainty ablations in this repo are simulation analogues of questions posed by WeatherPrompt and Ctrl-U — cite those papers if you build on them. Pointers: [`docs/literature_review.md`](docs/literature_review.md).

```bibtex
@inproceedings{zheng2020university,
  title     = {University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization},
  author    = {Zheng, Zhedong and Wei, Yunchao and Yang, Yi},
  booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
  year      = {2020}
}
```

Related methods: LPN, FSRA, MuSe-Net, WeatherPrompt, Ctrl-U, MC-Dropout.

## License

[MIT](LICENSE). Copyright (c) 2026 Magne Dina Neves.

---

[github.com/KRYPTON0078](https://github.com/KRYPTON0078)
