from __future__ import annotations

import numpy as np
import pandas as pd


def maskcolumns(
  df: pd.DataFrame,
  columns: list[str] | None = None,
  ratio: float = 0.15,
  seed: int = 42,
  exclude: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
  blocked = set(exclude or [])
  targets = columns if columns is not None else [column for column in df.columns if column not in blocked]
  masked = df.copy()
  truth: dict[str, pd.Series] = {}
  rng = np.random.default_rng(seed)
  for column in targets :
    if column not in masked.columns :
      continue
    nrows = len(masked)
    nmask = max(1, int(nrows * ratio))
    indices = rng.choice(masked.index, size = nmask, replace = False)
    truth[column] = masked.loc[indices, column].copy()
    if masked[column].dtype == bool :
      masked[column] = masked[column].astype(object)
    masked.loc[indices, column] = np.nan
  return masked, truth
