from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

ColumnKind = Literal["numerical", "categorical"]
DistributionShape = Literal["skewed", "normal_uniform"]
MissingnessFlag = Literal["ok", "high_missing"]

HIGH_MISSING_THRESHOLD = 0.70
HIGH_CARDINALITY_THRESHOLD = 10
SKEW_THRESHOLD = 1.0


@dataclass(frozen = True)
class ColumnProfile:
    name: str
    kind: ColumnKind
    missingratio: float
    cardinality: int
    distributionshape: DistributionShape | None
    missingnessflag: MissingnessFlag
    modelname: str | None = None


def isnumerical(series: pd.Series) -> bool:
  dtype = series.dtype
  return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)


def iscategorical(series: pd.Series) -> bool:
  return pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series)


def missingratio(series: pd.Series) -> float:
  return float(series.isna().mean())


def cardinality(series: pd.Series) -> int:
  return int(series.dropna().nunique())


def distributionshape(series: pd.Series) -> DistributionShape:
  values = series.dropna().astype(float)
  if len(values) < 8 :
    return "normal_uniform"
  skew = float(stats.skew(values))
  if abs(skew) >= SKEW_THRESHOLD :
    return "skewed"
  return "normal_uniform"


def ishighcardinality(card: int) -> bool:
  return card > HIGH_CARDINALITY_THRESHOLD


def isdiscrete(series: pd.Series, card: int) -> bool:
  if not isnumerical(series) :
    return False
  if card > 20 :
    return False
  if pd.api.types.is_integer_dtype(series) :
    return True
  values = series.dropna().astype(float)
  return bool(np.allclose(values, np.round(values)))


def profilecolumn(name: str, series: pd.Series) -> ColumnProfile:
  ratio = missingratio(series)
  flag: MissingnessFlag = "high_missing" if ratio > HIGH_MISSING_THRESHOLD else "ok"
  if flag == "high_missing" :
    warnings.warn(
      f"Column '{name}' has {ratio:.1%} missing values (> {HIGH_MISSING_THRESHOLD:.0%}). "
      "Imputation reliability may be limited.",
      stacklevel = 2,
    )
  if isnumerical(series) :
    card = cardinality(series)
    shape = distributionshape(series)
    return ColumnProfile(name, "numerical", ratio, card, shape, flag)
  return ColumnProfile(
    name,
    "categorical",
    ratio,
    cardinality(series),
    None,
    flag,
  )


def profiledataframe(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, ColumnProfile]:
  targets = columns if columns is not None else list(df.columns)
  return {column : profilecolumn(column, df[column]) for column in targets if column in df.columns}


def featurecolumns(df: pd.DataFrame, target: str, exclude: list[str] | None = None) -> list[str]:
  blocked = {target, *(exclude or [])}
  return [column for column in df.columns if column not in blocked]


def selectfeatures(
  df: pd.DataFrame,
  target: str,
  exclude: list[str] | None = None,
  topk: int = 6,
) -> list[str]:
  from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

  features = featurecolumns(df, target, exclude)
  if len(features) <= topk :
    return features
  observed = df[target].notna() & df[features].notna().all(axis = 1)
  if observed.sum() < 20 :
    return features
  xframe = df.loc[observed, features].copy()
  yseries = df.loc[observed, target]
  if len(xframe) > 8000 :
    xframe = xframe.sample(8000, random_state = 42)
    yseries = yseries.loc[xframe.index]
  xnum = pd.DataFrame({
    column : (
      xframe[column].astype(float)
      if isnumerical(xframe[column])
      else pd.Categorical(xframe[column]).codes
    )
    for column in features
  })
  if isnumerical(yseries) :
    scores = mutual_info_regression(xnum, yseries.astype(float), random_state = 42)
  else :
    scores = mutual_info_classif(xnum, yseries.astype(str), random_state = 42)
  ranked = sorted(zip(features, scores), key = lambda item : item[1], reverse = True)
  return [column for column, score in ranked[:topk] if score > 0] or [column for column, _ in ranked[:topk]]


def expandfeatures(
  df: pd.DataFrame,
  columns: list[str],
  dummycolumns: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
  parts: list[pd.DataFrame] = []
  dummymap: dict[str, list[str]] = {}
  for column in columns :
    if column not in df.columns :
      continue
    if isnumerical(df[column]) :
      parts.append(df[[column]].astype(float).rename(columns = {column : column}))
      continue
    dummies = pd.get_dummies(df[column].astype(str), prefix = column, dtype = float)
    if dummycolumns and column in dummycolumns :
      for name in dummycolumns[column] :
        if name not in dummies.columns :
          dummies[name] = 0.0
      dummies = dummies[dummycolumns[column]]
    else :
      dummymap[column] = list(dummies.columns)
    parts.append(dummies)
  if not parts :
    return pd.DataFrame(index = df.index), dummymap
  return pd.concat(parts, axis = 1), dummymap


def encodeframe(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, dict[object, int]]]:
  encoded = df.copy()
  maps: dict[str, dict[object, int]] = {}
  for column in columns :
    if column not in encoded.columns :
      continue
    series = encoded[column]
    if isnumerical(series) :
      encoded[column] = series.astype(float)
      continue
    labels = sorted(series.dropna().unique(), key = lambda value : str(value))
    mapping = {label : index for index, label in enumerate(labels)}
    maps[column] = mapping
    encoded[column] = series.map(mapping)
  return encoded, maps


def decodecolumn(series: pd.Series, mapping: dict[object, int]) -> pd.Series:
  inverse = {code : label for label, code in mapping.items()}
  return series.map(inverse)
