from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from simpute.models import buildmodel, pickmodel
from simpute.utils import (
    ColumnProfile,
    decodecolumn,
    expandfeatures,
    featurecolumns,
    isnumerical,
    profilecolumn,
    profiledataframe,
    selectfeatures,
)


class Simpute(BaseEstimator, TransformerMixin):
  """Adaptive per-column imputer with automatic model selection."""

  def __init__(
    self,
    columns: list[str] | None = None,
    exclude: list[str] | None = None,
    maskratio: float = 0.0,
    randomstate: int = 42,
  ) -> None:
    self.columns = columns
    self.exclude = exclude or []
    self.maskratio = maskratio
    self.randomstate = randomstate
    self.profiles_: dict[str, ColumnProfile] = {}
    self.models_: dict[str, Any] = {}
    self.featuremaps_: dict[str, list[str]] = {}
    self.dummycolumns_: dict[str, dict[str, list[str]]] = {}
    self.targetencodings_: dict[str, dict[object, int]] = {}
    self.featurefills_: dict[str, dict[str, float | str | bool]] = {}
    self.fallbacks_: dict[str, float | object] = {}
    self.booltargets_: set[str] = set()
    self.fittedcolumns_: list[str] = []

  def _targets(self, df: pd.DataFrame) -> list[str]:
    if self.columns is not None :
      return [column for column in self.columns if column in df.columns]
    return [column for column in df.columns if column not in self.exclude]

  def _columnorder(self, df: pd.DataFrame, columns: list[str]) -> list[str]:
    return sorted(
      columns,
      key = lambda column : (
        0 if self.profiles_.get(column) and self.profiles_[column].kind == "numerical" else 1,
        int(df[column].isna().sum()),
        column,
      ),
    )

  def _preparefeatures(self, df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    dummymap = self.dummycolumns_.get(target)
    expanded, learned = expandfeatures(df[features], features, dummymap)
    if target not in self.dummycolumns_ :
      self.dummycolumns_[target] = learned
    return expanded.astype(float)

  def _preparetarget(self, series: pd.Series, target: str) -> pd.Series:
    if isnumerical(series) :
      return series.astype(float)
    labels = sorted(series.dropna().unique(), key = lambda value : str(value))
    mapping = {label : index for index, label in enumerate(labels)}
    self.targetencodings_[target] = mapping
    return series.map(mapping).astype(float)

  def _fallbackvalue(self, series: pd.Series) -> float | object:
    observed = series.dropna()
    if observed.empty :
      return np.nan
    if isnumerical(series) :
      return float(observed.median())
    modes = observed.mode()
    return modes.iloc[0] if not modes.empty else observed.iloc[0]

  def _nativemissing(self, modelname: str) -> bool:
    return modelname in {"LGBMRegressor", "LGBMClassifier", "CatBoostClassifier"}

  def _featurefills(self, df: pd.DataFrame, features: list[str]) -> dict[str, float | str | bool]:
    fills: dict[str, float | str | bool] = {}
    for feature in features :
      series = df[feature]
      if isnumerical(series) :
        fills[feature] = float(series.dropna().median())
        continue
      mode = series.dropna().mode()
      fills[feature] = mode.iloc[0] if not mode.empty else series.dropna().iloc[0]
    return fills

  def _resetcolumn(self, target: str) -> None:
    self.models_.pop(target, None)
    self.featuremaps_.pop(target, None)
    self.dummycolumns_.pop(target, None)
    self.targetencodings_.pop(target, None)
    self.featurefills_.pop(target, None)
    self.fallbacks_.pop(target, None)
    self.booltargets_.discard(target)

  def _fitcolumn(self, data: pd.DataFrame, target: str) -> bool:
    self._resetcolumn(target)
    profile = self.profiles_.get(target) or profilecolumn(target, data[target])
    self.profiles_[target] = profile
    if profile.missingnessflag == "high_missing" and data[target].isna().all() :
      return False
    features = selectfeatures(data, target, self.exclude, topk = 6)
    if not features :
      return False
    observed = data[target].notna()
    if observed.sum() < 2 :
      return False

    trainframe = data.loc[observed].copy()
    fills = self._featurefills(trainframe, features)
    selectionframe = trainframe.copy()
    for feature, fill in fills.items() :
      selectionframe[feature] = selectionframe[feature].fillna(fill)

    xselect = self._preparefeatures(selectionframe, target, features).fillna(0.0)
    ytrain = self._preparetarget(trainframe[target], target)
    valid = xselect.notna().all(axis = 1) & ytrain.notna()
    xselect = xselect.loc[valid]
    ytrain = ytrain.loc[valid]
    if len(xselect) < 2 :
      return False

    modelname = pickmodel(profile, xselect.values, ytrain.values, data[target])
    usenative = self._nativemissing(modelname)

    if usenative :
      xtrain = self._preparefeatures(trainframe.loc[ytrain.index], target, features)
    else :
      filled = trainframe.loc[ytrain.index, features + [target]].copy()
      for feature, fill in fills.items() :
        filled[feature] = filled[feature].fillna(fill)
      xtrain = self._preparefeatures(filled, target, features).fillna(0.0)
      complete = xtrain.notna().all(axis = 1)
      xtrain = xtrain.loc[complete]
      ytrain = ytrain.loc[complete.index[complete]]

    if len(xtrain) > 15000 :
      keep = np.random.default_rng(self.randomstate).choice(len(xtrain), 15000, replace = False)
      xtrain = xtrain.iloc[keep]
      ytrain = ytrain.iloc[keep]

    if len(xtrain) < 2 :
      return False

    self.profiles_[target] = ColumnProfile(
      profile.name,
      profile.kind,
      profile.missingratio,
      profile.cardinality,
      profile.distributionshape,
      profile.missingnessflag,
      modelname,
    )
    model = buildmodel(modelname, self.profiles_[target], len(xtrain), xtrain.shape[1])
    model.fit(xtrain.values, ytrain.values)
    self.models_[target] = model
    self.featuremaps_[target] = features
    self.featurefills_[target] = fills
    self.fallbacks_[target] = self._fallbackvalue(data[target])
    if not hasattr(self, "usenative_") :
      self.usenative_ = {}
    self.usenative_[target] = usenative
    if pd.api.types.is_bool_dtype(data[target]) :
      self.booltargets_.add(target)
    return True

  def _predictvalues(self, model: Any, xpred: pd.DataFrame) -> np.ndarray:
    preds = np.asarray(model.predict(xpred.values))
    return preds.ravel()

  def _imputecolumn(self, result: pd.DataFrame, target: str) -> None:
    if target not in self.models_ :
      missing = result[target].isna()
      if missing.any() :
        result.loc[missing, target] = self._fallbackvalue(result[target])
      return

    missing = result[target].isna()
    if not missing.any() :
      return

    features = self.featuremaps_[target]
    block = result.loc[missing, features].copy()
    usenative = getattr(self, "usenative_", {}).get(target, False)
    if not usenative :
      for feature, fill in self.featurefills_[target].items() :
        block[feature] = block[feature].fillna(fill)

    xpred = self._preparefeatures(block, target, features)
    if not usenative :
      xpred = xpred.fillna(0.0)

    preds = self._predictvalues(self.models_[target], xpred)
    profile = self.profiles_[target]
    if profile.kind == "categorical" :
      decoded = decodecolumn(pd.Series(preds), self.targetencodings_[target])
      if target in self.booltargets_ :
        result.loc[missing, target] = decoded.astype(bool).values
      else :
        result.loc[missing, target] = decoded.values
    else :
      if profile.cardinality <= 20 :
        preds = np.round(preds)
      result.loc[missing, target] = np.asarray(preds, dtype = float)

    stillmissing = result[target].isna()
    if stillmissing.any() :
      result.loc[stillmissing, target] = self.fallbacks_[target]

  def fit(self, df: pd.DataFrame, y: Any = None) -> Simpute:
    del y
    data = df.copy()
    targets = self._targets(data)
    self.profiles_ = profiledataframe(data, targets)
    self.models_.clear()
    self.featuremaps_.clear()
    self.dummycolumns_.clear()
    self.targetencodings_.clear()
    self.featurefills_.clear()
    self.fallbacks_.clear()
    self.booltargets_.clear()

    order = self._columnorder(data, targets)
    for target in order :
      self._fitcolumn(data, target)

    self.fittedcolumns_ = list(self.models_.keys())
    return self

  def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    if not self.models_ :
      raise RuntimeError("Simpute is not fitted. Call fit before transform.")
    result = df.copy()
    order = self._columnorder(result, self.fittedcolumns_)
    for _ in range(3) :
      before = result.isna().sum().sum()
      for target in order :
        self._imputecolumn(result, target)
      if result.isna().sum().sum() == before :
        break
    for column in self._targets(result) :
      if column not in self.models_ and result[column].isna().any() :
        result[column] = result[column].fillna(self._fallbackvalue(df[column]))
    return result

  def fit_transform(self, df: pd.DataFrame, y: Any = None) -> pd.DataFrame:
    del y
    data = df.copy()
    targets = self._targets(data)
    self.profiles_ = profiledataframe(data, targets)
    self.models_.clear()
    self.featuremaps_.clear()
    self.dummycolumns_.clear()
    self.targetencodings_.clear()
    self.featurefills_.clear()
    self.fallbacks_.clear()
    self.booltargets_.clear()

    result = data.copy()
    order = self._columnorder(result, targets)
    for target in order :
      if result[target].isna().sum() == 0 :
        continue
      self._fitcolumn(result, target)
      self._imputecolumn(result, target)

    self.fittedcolumns_ = list(self.models_.keys())
    return result

  def getprofiles(self) -> dict[str, ColumnProfile]:
    return copy.deepcopy(self.profiles_)

  def getmodelselection(self) -> dict[str, str]:
    return {
      target : profile.modelname
      for target, profile in self.profiles_.items()
      if profile.modelname is not None
    }
