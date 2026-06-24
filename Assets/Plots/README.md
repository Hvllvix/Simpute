# Validation Plots

Release validation figures for the bundled test dataset. Committed to the repo so they render on GitHub without running anything locally.

| File | Description |
|------|-------------|
| `imputation_density.png` | KDE of observed vs post-imputation continuous distributions |
| `missingness_heatmap.png` | Feature completeness before and after imputation |
| `model_allocation_grid.png` | Model backend assigned per column |

Regenerate after changing `tests/data/test.csv`:

```bash
python scripts/generate_plots.py
```

Settings: `MASKRATIO=0.15`, `SEED=42`, dark theme (`#22272E`).
