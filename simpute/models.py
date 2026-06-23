from __future__ import annotations

from typing import Any

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier as LGBMC, LGBMRegressor as LGBMR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from simpute.utils import ColumnProfile, isdiscrete, ishighcardinality


def _lgbmparams(nrows: int) -> dict[str, Any]:
  leaves = min(63, max(15, int(np.sqrt(nrows))))
  estimators = min(150, max(50, nrows // 150))
  return {
    "n_estimators" : estimators,
    "num_leaves" : leaves,
    "learning_rate" : 0.05,
    "random_state" : 42,
    "verbosity" : -1,
    "n_jobs" : -1,
  }


def candidates(profile: ColumnProfile, nrows: int, series: Any = None) -> list[str]:
  if profile.kind == "categorical" :
    if ishighcardinality(profile.cardinality) :
      return ["CatBoostClassifier", "LGBMClassifier"]
    if profile.cardinality <= 2 :
      return ["LogisticRegression"]
    return ["LogisticRegression", "LinearSVC"]
  discrete = series is not None and isdiscrete(series, profile.cardinality)
  if nrows >= 1000 :
    return ["LGBMRegressor", "ExtraTreesRegressor"]
  if discrete or profile.distributionshape == "skewed" :
    return ["LGBMRegressor", "ExtraTreesRegressor"]
  return ["KNNRegressor", "BayesianRidge"]


def selectmodel(profile: ColumnProfile, nrows: int, series: Any = None) -> str:
  return candidates(profile, nrows, series)[0]


def buildmodel(modelname: str, profile: ColumnProfile, nrows: int, nfeatures: int) -> Any:
  if modelname == "LGBMClassifier" :
    return LGBMC(**_lgbmparams(nrows))
  if modelname == "CatBoostClassifier" :
    return CatBoostClassifier(
      iterations = min(300, max(100, nrows // 50)),
      depth = min(8, max(4, int(np.log2(nrows + 1)))),
      learning_rate = 0.05,
      random_seed = 42,
      verbose = False,
      thread_count = -1,
    )
  if modelname == "LogisticRegression" :
    return Pipeline([
      ("scaler", StandardScaler()),
      ("model", LogisticRegression(
        max_iter = 2000,
        C = 1.0,
        class_weight = "balanced",
        random_state = 42,
      )),
    ])
  if modelname == "LinearSVC" :
    return Pipeline([
      ("scaler", StandardScaler()),
      ("model", LinearSVC(max_iter = 3000, class_weight = "balanced", random_state = 42)),
    ])
  if modelname == "LGBMRegressor" :
    return LGBMR(**_lgbmparams(nrows))
  if modelname == "ExtraTreesRegressor" :
    return ExtraTreesRegressor(
      n_estimators = min(300, max(100, nrows // 50)),
      max_features = "sqrt",
      random_state = 42,
      n_jobs = -1,
    )
  if modelname == "KNNRegressor" :
    neighbors = min(50, max(5, int(np.sqrt(nrows))))
    return Pipeline([
      ("scaler", StandardScaler()),
      ("model", KNeighborsRegressor(n_neighbors = neighbors, weights = "distance", n_jobs = -1)),
    ])
  if modelname == "BayesianRidge" :
    return Pipeline([
      ("scaler", StandardScaler()),
      ("model", BayesianRidge()),
    ])
  raise ValueError(f"Unsupported model: {modelname}")


def pickmodel(
  profile: ColumnProfile,
  xtrain: np.ndarray,
  ytrain: np.ndarray,
  series: Any = None,
) -> str:
  options = candidates(profile, len(ytrain), series)
  if len(options) == 1 :
    return options[0]
  if profile.kind == "numerical" and len(ytrain) >= 1000 :
    return "LGBMRegressor" if "LGBMRegressor" in options else options[0]
  if profile.kind == "categorical" and len(ytrain) >= 1000 and "CatBoostClassifier" in options :
    return "CatBoostClassifier"
  if profile.kind == "categorical" and len(ytrain) >= 1000 :
    return options[0]
  return options[0]
