# Simulation study: weather robustness and uncertainty-aware scoring for drone→satellite retrieval

**Magne Dina Neves** · University of Macau · synthetic campus only · seed 42 · CPU TinyCNN  
Companion: [literature review](../docs/literature_review.md) · raw numbers: [`metrics.json`](metrics.json)

---

## Abstract

We study cross-view geo-localization as **labeled-nuisance retrieval** on a procedural campus (12 train / 8 test locations, 32 drone queries, 8 satellite gallery tiles). A shared TinyCNN dual-view encoder is trained with InfoNCE + identity loss. Two recipes are compared: **clean training** vs **WeatherPrompt-style weather augmentation** (fog/rain/night/snow). At test time we add geometric nuisances (occlusion, yaw, altitude) and an **uncertainty-aware (UA)** ranker: Monte-Carlo dropout on the query, cosine scores multiplied by \(1/(1+\sigma)\).

On this split, clean training reaches **R@1 = 0.719 / mAP = 0.817** on clean queries (`best.pt`). Fog and night collapse that model to **R@1 = 0.125**. Weather-aug **does not** match clean-set accuracy (R@1 0.594) but **raises night R@1 from 0.125 to 0.281** and fog from 0.125 to 0.156. UA scoring helps occlusion for the weather-aug model (0.469 → 0.625) and is mixed elsewhere. **These numbers are not University-1652.** They are a CPU-reproducible existence proof of the tradeoff the literature describes.

## 1. Introduction

Smart-city UAV stacks still fail the simple question: *which satellite tile is this drone looking at?* GNSS dropout, facade-like buildings, and Macau-style rain/fog make the match brittle. University-1652 ([Zheng, Wei, Yang, ACM MM 2020](https://doi.org/10.1145/3394171.3413896)) is the right exam; it is also multi-gigabyte and gated. This study answers a prior question that a 4th-year student can actually run on a laptop:

> If location identity is known and nuisances are *labeled*, which training and scoring choices survive weather and geometry, and what do they cost on clean data?

That is the experimental style of WeatherPrompt and MuSe-Net (robustness) and Ctrl-U (variance as an uncertainty instrument), executed as a **simulation**, not as a fake leaderboard.

## 2. Related work

See [`docs/literature_review.md`](../docs/literature_review.md). Short version: ID-loss CNNs → LPN partitions → FSRA transformers on U1652; weather treated as domain shift (MuSe-Net) or language-gated (WeatherPrompt); uncertainty via MC-Dropout / reward variance (Ctrl-U). We implement the cheap cousins: classical weather overlays + MC-Dropout reweighting.

## 3. Method

![Architecture](../figures/architecture.png)

- **Simulation.** Each building has a unique hue/footprint/roof and a pose \((x,y)\) on a 20 m grid. Satellite tiles are nadir, north-up, 500 m AGL. Drone tiles record altitude (70–106 m) and yaw. Generator: `python -m aerial_geoloc.cli sim-generate`.
- **Encoder.** Shared TinyCNN + MLP projection, L2-normalized 64-D embeddings, dropout 0.15 (needed for MC-Dropout).
- **Loss.** Symmetric in-batch InfoNCE (\(\tau=0.07\)) + 0.5× identity CE.
- **Weather-aug.** Train-time RandomWeather with \(p=0.7\) over `{fog, rain, night, snow}`.
- **UA scoring.** \(T=6\) dropout samples on the **query only**; gallery is deterministic. \(\mathrm{score} = \cos(z_q, z_g)\,/\,(1+\sigma_q)\).

## 4. Experimental setup

| Item | Value |
| --- | --- |
| Data | `data/sim/` (procedural; **not** U1652 imagery) |
| Train locations / test locations | 12 / 8 |
| Queries / gallery | 32 drone / 8 satellite |
| Image size | 64×64 |
| Epochs / batch / seed / device | 6 / 16 / 42 / CPU |
| Checkpoint used for condition tables | `outputs/study_*/best.pt` (best clean-test R@1) |
| Conditions | clean, fog, rain, night, snow, occlusion, yaw, altitude |

Reproduce everything on CPU (minutes, usually < 1):

```bash
pip install -e ".[dev,study]"
python -m aerial_geoloc.cli study --config configs/study.yaml --regenerate
pytest
```

## 5. Results

All figures are generated from [`metrics.json`](metrics.json). No cells were typed by hand.

### 5.1 Training dynamics

![Training curves](../figures/training_curves.png)

Clean training reaches **R@1 0.719** at epoch 3 then overfits slightly (last epoch 0.594). Weather-aug starts from a harder loss and plateaus lower on the *clean* test set (best **0.594**). That is the expected robustness tax, not a bug.

### 5.2 Conditioned retrieval

**Recall@1** (32 queries). `UA` = weather-aug model + uncertainty-aware scores.

| Condition | Baseline cosine | Weather-aug cosine | Weather-aug + UA |
| --- | ---: | ---: | ---: |
| clean | **0.719** | 0.594 | 0.531 |
| fog | 0.125 | 0.156 | **0.188** |
| rain | **0.625** | 0.562 | 0.500 |
| night | 0.125 | **0.281** | 0.188 |
| snow | **0.719** | 0.531 | 0.438 |
| occlusion | **0.688** | 0.469 | 0.625 |
| yaw | **0.656** | 0.469 | 0.469 |
| altitude | **0.562** | 0.406 | 0.312 |

**mAP** (cosine only):

| Condition | Baseline | Weather-aug |
| --- | ---: | ---: |
| clean | **0.817** | 0.725 |
| fog | 0.345 | **0.390** |
| rain | **0.770** | 0.701 |
| night | 0.351 | **0.514** |
| snow | **0.816** | 0.671 |
| occlusion | **0.786** | 0.679 |
| yaw | **0.818** | 0.653 |
| altitude | **0.753** | 0.622 |

![R@1 bars](../figures/recall_by_condition.png)

![mAP bars](../figures/map_by_condition.png)

**Reading the table without marketing:** weather-aug wins the two conditions that actually destroy the baseline (fog, night). It loses almost everywhere else, including rain and snow on this particular renderer (those overlays are milder than night/fog). UA is not a free lunch: it recovers occlusion for the weather-aug model and a bit of fog, and it *hurts* clean R@1 because dropout noise moves an already-peaked ranking.

### 5.3 Qualitative retrieval

![Fog retrieval grid](../figures/retrieval_grid_fog.png)

Fog queries (left) vs top-5 satellite tiles. Green = correct location ID. This grid is a *sample of queries*, not the 0.156 aggregate — several of the first queries happen to hit R@1 even when the split average is low.

![Night failures](../figures/failures_night.png)

Baseline failures under night: query | predicted satellite | true satellite.

### 5.4 Embedding geometry and uncertainty

![PCA](../figures/embedding_pca.png)

PCA of drone (circles) and satellite (stars) embeddings, colored by location. Overlap of the two markers per color is the geometric hope of the dual-view encoder.

![Uncertainty vs recall](../figures/uncertainty_vs_recall.png)

Mean MC-Dropout \(\sigma\) vs R@1 across conditions for the weather-aug model. Night/fog are both high-error; \(\sigma\) does **not** form a perfect calibration curve on this tiny encoder — we report that rather than forcing a story.

## 6. Discussion

1. **The literature tradeoff is real even in a toy world.** Clean specialists die in fog/night (R@1 0.125). Weather-aug pays on clean data to buy night R@1 0.281. That is the same qualitative message as MuSe-Net / WeatherPrompt, at a scale a CPU can plot.
2. **UA scoring is a scalpel, not a backbone.** Ctrl-U down-weights noisy rewards; we down-weight noisy *queries*. When the embedding is already wrong, \(\sigma\) does not invent the right satellite. When occlusion makes dropout disagree, reweighting can help (0.469 → 0.625).
3. **Snow/rain on this renderer are weak nuisances.** Baseline snow R@1 matches clean (0.719). Do not read that as “the model is snow-robust in the field”; the overlay is too light. Night and fog are the stress tests that actually move pixels.

## 7. Limitations

- **Synthetic identity is color+shape**, not architecture or roof texture from Google Earth. A TinyCNN can cheat in ways that fail on U1652.
- 8 gallery identities is a small closed set. Rank-5 saturates (often 0.94–1.0); R@1 is the informative column.
- 6 epochs, no hyperparameter search, no ResNet-50, no LPN partitions, no VLM weather text.
- Checkpoints themselves are not in git (`*.pt` ignored); metrics and figures are.

## 8. Future work (path to University-1652)

1. Point `configs/default.yaml` at a local `University-Release/` tree ([docs/DATASETS.md](../docs/DATASETS.md)).
2. Swap `tiny_cnn` → `resnet18` with ImageNet init (outside CI).
3. Re-run the same eight conditions as **eval overlays** on real U1652 drone queries (WeatherPrompt-style), not as a claim that we trained on U1652-WX.
4. Replace MC-Dropout with a proper evidential or Gaussian embedding head if \(\sigma\) remains poorly calibrated.
5. Do not paste LPN/FSRA/WeatherPrompt tables into this file unless those runs are executed here.

## 9. How to reproduce

```bash
python -m aerial_geoloc.cli sim-generate --root data/sim
python -m aerial_geoloc.cli study --config configs/study.yaml
# writes reports/metrics.json and figures/*.png
```

Seed 42, CPU, `configs/study.yaml`. Re-runs can move a few queries; they still will not be University-1652.
