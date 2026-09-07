# Related work: aerial geo-localization, weather shift, and uncertainty

This note is the literature spine for `aerial-spatial-geoloc`. It is written for a researcher who already knows image retrieval, not as a blog post. **None of the numbers below are results of this repository**; they are attributed to the cited papers. Our measurements live in [`reports/EXPERIMENT_REPORT.md`](../reports/EXPERIMENT_REPORT.md) and are synthetic-only.

## 1. Problem statement

Cross-view geo-localization (CVGL) asks: given an image from platform A, retrieve the matching geo-referenced image from platform B. When A is a UAV and B is a satellite, the viewpoint gap is extreme — oblique vs. nadir, different sensors, different season and weather. Zheng, Wei, and Yang argued that drones are a *third* platform between phones and satellites, and built a benchmark around that claim.

Two operational tasks follow ([Zheng et al., ACM MM 2020](https://doi.org/10.1145/3394171.3413896)):

1. **Drone-view target localization** (drone → satellite): “where on the map is this UAV frame?”
2. **Drone navigation** (satellite → drone): “which flight snippet matches this satellite crop?”

Both are instance-level retrieval with location IDs as the class, not pixel-wise registration.

## 2. Benchmarks

| Dataset | Platforms | What it contributes | Citation |
| --- | --- | --- | --- |
| **University-1652** | drone, satellite, street; 1,652 buildings / 72 universities | First large drone-based CVGL set; still the default protocol for UAV↔sat | [Zheng, Wei, Yang, ACM MM 2020](https://doi.org/10.1145/3394171.3413896); [arXiv:2002.12186](https://arxiv.org/abs/2002.12186); [baseline repo](https://github.com/layumi/University1652-Baseline) |
| **CVUSA / CVACT** | ground panorama ↔ satellite | Street-to-sat; orientation and polar transforms matter | Workman et al.; Liu & Li |
| **SUES-200** | real drone at multiple heights ↔ satellite | Height as a controlled nuisance | Zhu et al. |
| **University-1652-WX / weather extensions** | U1652 + synthesized weather | Stress tests used by WeatherPrompt and MuSe-Net | see §4 |

University-1652 imagery originates from Google Maps/Earth and is **non-commercial academic**. This toolkit adapters the *folder contract* and does not redistribute pixels.

Reported baseline numbers on U1652 (literature, **not this repo**):

- Zheng et al. (2020) instance-loss CNN: drone→sat R@1 around the low-70s in the original paper’s ResNet setting (see the paper’s Table 2; do not copy into our result table).
- LPN ([Wang, Zheng, Yan, Yang, IEEE TCSVT 2021](https://doi.org/10.1109/TCSVT.2021.3061265)): square-ring partitions; drone→sat R@1 **86.45**, AP **74.79** as listed on the [official SOTA page](https://github.com/layumi/University1652-Baseline/tree/master/State-of-the-art).
- FSRA (Dai et al., TCSVT 2022): ViT + region alignment; higher still (k=1 R@1 **87.87** / AP **81.53** on the same page).

Those figures are **published leaderboard facts**. Filling our README with them as if we ran U1652 would be dishonest.

## 3. Methods: from ID loss to contrastive dual encoders

Early U1652 baselines treat each building as an identity (classification + triplet), in the person-re-ID tradition Zheng helped define. Subsequent work adds:

- **Spatial partitions**: LPN’s square-ring pooling ([Wang et al., 2021](https://doi.org/10.1109/TCSVT.2021.3061265)) so context around the building is not discarded.
- **Transformers**: FSRA and later ViT geo-localizers, aligning regions across views.
- **Contrastive dual encoders**: InfoNCE / CLIP-style in-batch negatives ([Oord et al., CPC](https://arxiv.org/abs/1807.03748); [Radford et al., CLIP](https://arxiv.org/abs/2103.00020)) — the loss this toolkit uses by default. It is a modern cousin of the original ID loss, not a claim of novelty.

SAFA (Shi et al.) and polar-transform methods matter more for ground↔sat (CVUSA) than for UAV nadir, but the shared idea is: *canonicalise geometry before matching*.

## 4. Weather and domain shift

Clean-weather U1652 overstates field performance. Two lines of work from Zheng’s circle are directly relevant:

- **MuSe-Net** — Wang, Zheng, Sun, Yan, Yang, Chua, *Pattern Recognition* 2024. [doi:10.1016/j.patcog.2024.110363](https://doi.org/10.1016/j.patcog.2024.110363). Models environmental style as a domain and adapts features when drone appearance shifts while satellites stay put.
- **WeatherPrompt** — Wen, Yu, Zheng, NeurIPS 2025. [arXiv:2508.09560](https://arxiv.org/abs/2508.09560); [code](https://github.com/Jahawn-Wen/WeatherPrompt). Training-free weather *text* from a VLM, then gated fusion of image and weather language so scene identity is not entangled with rain/fog/night. They report large R@1 lifts under night and fog on U1652/SUES-200.

This repository does **not** reimplement WeatherPrompt’s VLM gating. It takes the cheaper experimental question: *if we can label the nuisance in simulation (fog, rain, night, snow, occlusion, yaw, altitude), does weather-aug training and uncertainty-aware scoring recover Recall@1?* That is a stepping stone, not a substitute.

Classical corruptions (fog as additive haze, rain streaks, night gamma) are a known stress test in robust vision (ImageNet-C, Hendrycks & Dietterich). They are not photoreal weather. WeatherPrompt’s point is exactly that categorical weather labels oversimplify; we still use them because they are *controllable* in a CPU sim.

## 5. Uncertainty

Zheng’s broader agenda includes **uncertainty estimation** (see also Ctrl-U).

- **MC-Dropout** — Gal & Ghahramani, ICML 2016: dropout at test time as a cheap posterior; variance across samples is an uncertainty signal.
- **Ctrl-U** — Zhang, Gao, Zhao, Zheng, ICLR 2025. [OpenReview / ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/d76f8ea185a09f1dbb57a3365a4af768-Paper-Conference.pdf). Not a geo-loc paper: it uses **prediction variance of a reward model** to down-weight unreliable feedback in conditional generation. The transferable idea is: *variance is a usable uncertainty instrument; low-uncertainty evidence should count more.*

We apply that idea to retrieval: keep the gallery deterministic, sample the query encoder with dropout, and multiply cosine scores by \(1/(1+\sigma)\). That is an ablation, not Ctrl-U.

Related retrieval-uncertainty work (evidential matching, probabilistic embeddings, GeoMatch’s semantic uncertainty on limited-FoV CVGL) exists; we cite it as future reading rather than claiming we implemented it.

## 6. Aerial spatial intelligence and smart cities

Zheng’s AIGC-DL Lab frames UAV vision as **spatial intelligence** for the low-altitude economy: last-meter delivery, inspection, GNSS-denied “where am I on the map?” University-1652 is the data-centric bet that a third view (drone) makes those systems learnable. WeatherPrompt is the robustness bet that the same matcher must survive Macau rain and night. Ctrl-U is the uncertainty bet that models should know when not to trust themselves.

A portfolio project that *only* trains a CNN on a toy split does not speak that language. A project that (i) states the retrieval protocol, (ii) simulates labeled nuisances, (iii) measures the weather/clean tradeoff, and (iv) is honest that U1652 is still the real exam — does.

## 7. What this toolkit claims vs. does not claim

| Claim | Status |
| --- | --- |
| Runnable dual-view InfoNCE baseline with a U1652 adapter | yes |
| Simulation with known pose + labeled nuisances | yes |
| Weather-aug and MC-Dropout UA scoring as CPU ablations | yes |
| Reimplementation of WeatherPrompt / LPN / FSRA / Ctrl-U | **no** |
| University-1652 SOTA numbers from this repo | **no — not measured here** |

## 中文摘要

跨视角地理定位把无人机图像检索到卫星底图，是低空经济与智慧城市的基础能力。University-1652（Zheng et al., ACM MM 2020）定义了 drone↔satellite 协议；LPN / FSRA 等后续工作提高了干净天气下的 Recall。天气域移（MuSe-Net、WeatherPrompt）和不确定性（MC-Dropout、Ctrl-U 的方差加权思想）是同一实验室线索的延伸。本仓库用**可控仿真**测量天气增强与不确定性加权的消融，**不**把文献中的 U1652 数字写成自己的结果，真实数据实验仍待在 University-1652 上完成。

## References (selected)

1. Z. Zheng, Y. Wei, Y. Yang. University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization. ACM MM 2020. https://doi.org/10.1145/3394171.3413896
2. T. Wang, Z. Zheng, C. Yan, J. Zhang, Y. Sun, B. Zheng, Y. Yang. Each Part Matters: Local Patterns Facilitate Cross-View Geo-Localization. IEEE TCSVT, 2021. https://doi.org/10.1109/TCSVT.2021.3061265
3. M. Dai et al. A Transformer-Based Feature Segmentation and Region Alignment Method for UAV-View Geo-Localization. IEEE TCSVT, 2022.
4. T. Wang, Z. Zheng, Y. Sun, C. Yan, Y. Yang, T.-S. Chua. Multiple-environment Self-adaptive Network for Aerial-view Geo-localization. Pattern Recognition, 2024. https://doi.org/10.1016/j.patcog.2024.110363
5. J. Wen, H. Yu, Z. Zheng. WeatherPrompt: Multi-modality Representation Learning for All-Weather Drone Visual Geo-Localization. NeurIPS 2025. https://arxiv.org/abs/2508.09560
6. G. Zhang, H.-A. Gao, Z. Jiang, H. Zhao, Z. Zheng. Ctrl-U: Robust Conditional Image Generation via Uncertainty-aware Reward Modeling. ICLR 2025.
7. Y. Gal, Z. Ghahramani. Dropout as a Bayesian Approximation. ICML 2016.
8. A. van den Oord, Y. Li, O. Vinyals. Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748.
9. A. Radford et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021.
10. Official U1652 baseline and SOTA tables: https://github.com/layumi/University1652-Baseline
