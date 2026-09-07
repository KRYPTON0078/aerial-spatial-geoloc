"""University-1652 and related datasets.

University-1652 (Zheng, Wei, Yang, ACM MM 2020)
------------------------------------------------
Academic, **non-commercial** research use. Imagery originates from Google
Maps/Earth; follow the authors' Terms of Use. Cite the paper in any work
that uses the data, models trained on it, or numbers measured on it.

Obtain the release from:

* Baseline repo (dataset links + evaluation protocol):
  https://github.com/layumi/University1652-Baseline
* Hugging Face (gated):
  https://huggingface.co/datasets/layumi/university-1652

After extraction, point `data.root` at the folder that contains `train/` and
`test/` (often named `University-Release`):

```
<data.root>/
  train/
    drone/<building_id>/*.jpg
    satellite/<building_id>/*.jpg
    street/<building_id>/*.jpg
  test/
    query_drone/<building_id>/*.jpg
    query_satellite/<building_id>/*.jpg
    gallery_drone/<building_id>/*.jpg
    gallery_satellite/<building_id>/*.jpg
    gallery_street/<building_id>/*.jpg
```

Then:

```bash
python -m aerial_geoloc.cli train --config configs/default.yaml
python -m aerial_geoloc.cli eval --config configs/default.yaml --checkpoint outputs/university1652_resnet18/best.pt
```

Standard tasks
--------------
* **Drone → Satellite** (drone-view target localization): query `drone`, gallery `satellite`.
* **Satellite → Drone** (drone navigation): swap `data.query_view` / `data.gallery_view` in the YAML.

This toolkit does not redistribute University-1652 images.

Synthetic demo set
------------------
`python -m aerial_geoloc.cli make-demo-data` (also auto-generated on `train` when
`dataset: demo`) writes a **tiny geometric campus** with the same folder
convention. It is for pipeline smoke-tests only — not a substitute for U1652.

SUES-200
--------
See `aerial_geoloc.data.sues200` for a documented stub (used alongside U1652
in WeatherPrompt-style robustness papers). A full height-aware loader is not
implemented yet.
