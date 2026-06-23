# Simpute

Smart Impute (`Simpute`) is a production-ready Python package for adaptive missing-value imputation. Instead of applying one global method, it profiles each column and selects an optimal model from a curated suite of classifiers and regressors.

## Features

- Per-column data profiling: type detection, missingness ratio, cardinality, and distribution shape
- Automatic model selection across LightGBM, CatBoost, logistic models, SVM, KNN, Bayesian Ridge, and Extra Trees
- Programmatic warnings when a column exceeds 70% missing data
- Sklearn-compatible `fit`, `transform`, and `fit_transform` API
- Guard test suite that validates imputation guarantees against a ground-truth dataset

## Installation

```bash
pip install simpute
```

Development install:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from simpute import Simpute

df = pd.read_csv("data.csv")
imputer = Simpute(exclude=["id_column"])
filled = imputer.fit_transform(df)

print(imputer.getmodelselection())
```

## Model Selection Rules

| Column Profile | Candidate Models |
|----------------|------------------|
| High-cardinality categorical | LightGBM Classifier, CatBoost Classifier |
| Low-cardinality / binary categorical | Logistic Regression (L2), Linear SVC |
| Skewed continuous | LightGBM Regressor, Extra Trees Regressor |
| Normal / uniform continuous | KNN Regressor, Bayesian Ridge |

## Guard Tests

Run the zero-compromise verification suite:

```bash
pytest tests/guard.py -v
```

Guard tests mask values in `tests/data/test.csv`, impute them, and assert:

- No NaN values remain after imputation
- Categorical predictions stay within the original domain
- Numerical predictions stay within bounded ranges
- Imputation accuracy meets minimum thresholds against ground truth
- Model selection is deterministic and profile-consistent

## License

MIT
