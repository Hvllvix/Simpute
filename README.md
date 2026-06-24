# Smart Imputation

**Simpute** is an adaptive missing-value imputation library for tabular data. Instead of applying one global strategy to every column, it profiles each feature, selects a tailored model, and imputes columns sequentially so earlier fills inform later ones.

Install from PyPI as `simpute`. Source and releases live at [github.com/Hvllvix/Simpute](https://github.com/Hvllvix/Simpute).

---

## Why Simpute

Most imputers pick a single method (mean, median, MICE, KNN) for the whole table. Real datasets mix binary flags, low-cardinality categories, high-cardinality text-like fields, skewed counts, and smooth continuous variables. Simpute treats each column on its own terms.

| Core Architectural Dimension | Simpute Engine Standard |
| :--- | :--- |
| **Profiling Strategy** | Granular per-column analysis and dynamic routing |
| **API Compliance** | Native Scikit-learn interface (`fit` / `transform` / `fit_transform`) |
| **Algorithmic Suite** | LightGBM, CatBoost, Regularized Logistic/SVC, KNN, Bayesian Ridge, Extra Trees |
| **System Integrity** | Integrated firewall guard suite featuring ground-truth precision verification |
| **Fault Tolerance** | Automated warnings and flags for columns exceeding 70% missingness thresholds |

---

## Installation

```bash
pip install simpute
```

Development install with tests and plotting extras:

```bash
git clone [https://github.com/Hvllvix/Simpute.git](https://github.com/Hvllvix/Simpute.git)
cd Simpute
pip install -e ".[dev]"
```

---

## Quick Start

```python
import pandas as pd
from simpute import Simpute

df = pd.read_csv("data.csv")

imputer = Simpute(exclude=["Student_ID"])
filled = imputer.fit_transform(df)

print(imputer.getmodelselection())
print(imputer.getprofiles())
```

`exclude` keeps identifier columns out of the imputation loop. Use `columns=[...]` instead when you only want to impute a subset.

---

## How It Works

1. **Profile** each target column (type, missingness, cardinality, distribution shape).
2. **Select features** with mutual information (top 6 predictors by default).
3. **Route** to a candidate model based on the column profile.
4. **Fit** on observed rows, then **impute** missing cells column by column.
5. **Warn** when missingness exceeds 70% on a column.

Sequential imputation means numerical columns are generally filled before categorical ones, and values imputed in earlier columns become features for later columns.

---

## Model Selection

| Target Column Profile | Underlying Statistical Property | Optimized Backend Algorithm |
| :--- | :--- | :--- |
| **High-Cardinality Categorical** | Large nominal domains, text-like properties | `CatBoostClassifier` / `LightGBMClassifier` |
| **Low-Cardinality / Binary** | Binary indicators, low unique nominal categories | `LogisticRegression` (L2) / `LinearSVC` |
| **Large Numerical Tables** | Datasets exceeding 1,000 observations | `LightGBMRegressor` / `ExtraTreesRegressor` |
| **Skewed / Discrete Numerical** | Long-tailed metrics, highly unbalanced distributions | `LightGBMRegressor` / `ExtraTreesRegressor` |
| **Normal / Uniform Continuous** | Symmetric, un-skewed numerical continuous shapes | `KNNRegressor` / `BayesianRidge` |

Inspect the chosen backend per column after fitting:

```python
imputer.getmodelselection()

# {'Pre_Semester_GPA': 'LGBMRegressor', 'Major_Category': 'CatBoostClassifier', ...}
```

---

## API Reference

| Interface Method | Return Signature | Functional Description |
| :--- | :--- | :--- |
| `fit(df)` | `self` | Profiles columns and trains tailored per-column machine learning architectures. |
| `transform(df)` | `pd.DataFrame` | Executes sequential imputation calculations using previously fitted backend models. |
| `fit_transform(df)` | `pd.DataFrame` | Runs profiling, model training, and cell imputation in a single optimized pass. |
| `getprofiles()` | `dict` | Exposes the underlying metadata mapping generated during the dataset profiling phase. |
| `getmodelselection()` | `dict` | Returns the specific machine learning model mapped to each target imputed column. |

Constructor options: `columns`, `exclude`, `maskratio`, `randomstate`.

---

## Guard Tests

The guard suite (`tests/guard.py`) masks values in [`tests/data/test.csv`](tests/data/test.csv), imputes them, and checks:

- No NaN values remain after imputation
- Categorical predictions stay within the original domain
- Numerical predictions stay within bounded ranges
- Imputation beats adaptive random baselines on held-out masked cells
- Model selection is deterministic and profile-consistent
- High-missingness columns emit warnings
- `transform` before `fit` raises `RuntimeError`

See [`tests/data/README.md`](tests/data/README.md) for column descriptions and how to swap in your own CSV.

```bash
pytest tests/guard.py -v
```

Metric summary table (MAE for continuous columns, accuracy for nominal):

```bash
python tests/guard.py
```

---

## Validation Plots

Generated on the bundled test dataset (`MASKRATIO=0.15`, `SEED=42`):

| Target Asset Graphic | Metric Visualization Type | Core Analytical Purpose |
| :--- | :--- | :--- |
| [Imputation Density](Assets/Plots/imputation_density.png) | Kernel Density Estimation (KDE) | Compares baseline vs post-imputation distributions to verify variance preservation. |
| [Missingness Heatmap](Assets/Plots/missingness_heatmap.png) | Binary Feature Completeness Grid | Displays visual evidence of structural integrity before and after complete table imputation. |
| [Model Allocation](Assets/Plots/model_allocation_grid.png) | Horizontal System Flow Chart | Provides full clarity into how columns were programmatically routed to distinct algorithms. |

Regenerate locally:

```bash
python scripts/generate_plots.py
```

---

## Requirements

- Python 3.10+
- NumPy, Pandas, SciPy, scikit-learn, LightGBM, CatBoost

---

## Contributing

1. Fork [Hvllvix/Simpute](https://github.com/Hvllvix/Simpute)
2. Create a branch, make changes, run `pytest tests/guard.py -v`
3. Open a pull request

---

## License

MIT
```
