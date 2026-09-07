Procedural campus used by `python -m aerial_geoloc.cli study`.

Not University-1652 imagery. Pose metadata is in `manifest.json` (altitude, yaw, metric grid).

Regenerate (overwrites this folder):

```bash
rm -rf data/sim
python -m aerial_geoloc.cli sim-generate --root data/sim
# or
python -m aerial_geoloc.cli study --config configs/study.yaml --regenerate
```
