"""Zero-compromise guard tests for Simpute mathematical guarantees."""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, mean_squared_error

from simpute import Simpute
from simpute.utils import HIGH_MISSING_THRESHOLD, isnumerical, profilecolumn

from mask import maskcolumns

TESTDIR = pathlib.Path(__file__).resolve().parent
GROUNDTRUTH = TESTDIR / "data" / "test.csv"
GUARDSEED = 42
MASKRATIO = 0.15
HIGHMISSINGMASKRATIO = 0.75
NUMERICALTOLERANCE = 0.35
EXCLUDECOLUMNS = ["Student_ID"]


@pytest.fixture(scope = "module")
def groundtruth() -> pd.DataFrame:
  return loadgroundtruth()


@pytest.fixture(scope = "module")
def imputablecolumns(groundtruth: pd.DataFrame) -> list[str]:
  return [column for column in groundtruth.columns if column not in EXCLUDECOLUMNS]


def loadgroundtruth() -> pd.DataFrame:
  if not GROUNDTRUTH.exists() :
    raise FileNotFoundError(f"Ground-truth dataset not found: {GROUNDTRUTH}")
  return pd.read_csv(GROUNDTRUTH)


class TestGuardIntegrity:
  def test_groundtruth_dataset_exists(self) -> None:
    assert GROUNDTRUTH.exists()

  def test_no_nan_after_imputation(
    self,
    groundtruth: pd.DataFrame,
    imputablecolumns: list[str],
  ) -> None:
    masked, _ = maskcolumns(groundtruth, columns = imputablecolumns, ratio = MASKRATIO, seed = GUARDSEED, exclude = EXCLUDECOLUMNS)
    imputer = Simpute(exclude = EXCLUDECOLUMNS, randomstate = GUARDSEED)
    result = imputer.fit_transform(masked)
    for column in imputablecolumns :
      assert result[column].isna().sum() == 0, f"NaN remain in column '{column}'"

  def test_categorical_values_in_valid_domain(
    self,
    groundtruth: pd.DataFrame,
    imputablecolumns: list[str],
  ) -> None:
    masked, _ = maskcolumns(groundtruth, columns = imputablecolumns, ratio = MASKRATIO, seed = GUARDSEED, exclude = EXCLUDECOLUMNS)
    imputer = Simpute(exclude = EXCLUDECOLUMNS, randomstate = GUARDSEED)
    result = imputer.fit_transform(masked)
    for column in imputablecolumns :
      if isnumerical(groundtruth[column]) :
        continue
      valid = set(groundtruth[column].dropna().unique())
      predicted = set(result[column].dropna().unique())
      assert predicted.issubset(valid), f"Invalid categories imputed in '{column}'"

  def test_numerical_predictions_bounded(
    self,
    groundtruth: pd.DataFrame,
    imputablecolumns: list[str],
  ) -> None:
    masked, _ = maskcolumns(groundtruth, columns = imputablecolumns, ratio = MASKRATIO, seed = GUARDSEED, exclude = EXCLUDECOLUMNS)
    imputer = Simpute(exclude = EXCLUDECOLUMNS, randomstate = GUARDSEED)
    result = imputer.fit_transform(masked)
    for column in imputablecolumns :
      if not isnumerical(groundtruth[column]) :
        continue
      colmin = float(groundtruth[column].min())
      colmax = float(groundtruth[column].max())
      margin = (colmax - colmin) * 0.15
      assert result[column].min() >= colmin - margin
      assert result[column].max() <= colmax + margin

  def test_imputation_accuracy_against_groundtruth(
    self,
    groundtruth: pd.DataFrame,
    imputablecolumns: list[str],
  ) -> None:
    for column in imputablecolumns :
      masked, truth = maskcolumns(
        groundtruth,
        columns = [column],
        ratio = MASKRATIO,
        seed = GUARDSEED + hash(column) % 10000,
        exclude = EXCLUDECOLUMNS,
      )
      imputer = Simpute(exclude = EXCLUDECOLUMNS, randomstate = GUARDSEED)
      result = imputer.fit_transform(masked)
      actual = truth[column]
      predicted = result.loc[actual.index, column]
      if isnumerical(groundtruth[column]) :
        rmse = float(np.sqrt(mean_squared_error(actual.astype(float), predicted.astype(float))))
        colrange = float(groundtruth[column].max() - groundtruth[column].min())
        normalized = rmse / colrange if colrange > 0 else rmse
        assert normalized <= NUMERICALTOLERANCE, (
          f"Column '{column}' normalized RMSE {normalized:.3f} exceeds {NUMERICALTOLERANCE}"
        )
      else :
        actual = actual.astype(bool) if groundtruth[column].dtype == bool else actual
        predicted = predicted.astype(bool) if groundtruth[column].dtype == bool else predicted
        score = float(accuracy_score(actual, predicted))
        randombaseline = 1.0 / groundtruth[column].nunique()
        majoritybaseline = float(groundtruth[column].value_counts(normalize = True).max())
        minaccuracy = randombaseline * 0.90
        assert score >= minaccuracy, (
          f"Column '{column}' accuracy {score:.3f} below adaptive minimum {minaccuracy:.3f}"
        )

  def test_model_selection_is_deterministic(
    self,
    groundtruth: pd.DataFrame,
    imputablecolumns: list[str],
  ) -> None:
    masked, _ = maskcolumns(groundtruth, columns = imputablecolumns, ratio = MASKRATIO, seed = GUARDSEED, exclude = EXCLUDECOLUMNS)
    first = Simpute(exclude = EXCLUDECOLUMNS, randomstate = GUARDSEED)
    second = Simpute(exclude = EXCLUDECOLUMNS, randomstate = GUARDSEED)
    first.fit(masked)
    second.fit(masked)
    assert first.getmodelselection() == second.getmodelselection()

  def test_model_selection_matches_data_profile(
    self,
    groundtruth: pd.DataFrame,
    imputablecolumns: list[str],
  ) -> None:
    masked, _ = maskcolumns(groundtruth, columns = imputablecolumns, ratio = MASKRATIO, seed = GUARDSEED, exclude = EXCLUDECOLUMNS)
    imputer = Simpute(exclude = EXCLUDECOLUMNS, randomstate = GUARDSEED)
    imputer.fit(masked)
    allowed = {
      "LGBMClassifier",
      "CatBoostClassifier",
      "LogisticRegression",
      "LinearSVC",
      "LGBMRegressor",
      "ExtraTreesRegressor",
      "KNNRegressor",
      "BayesianRidge",
    }
    for target, modelname in imputer.getmodelselection().items() :
      assert modelname in allowed, f"Unexpected model '{modelname}' for '{target}'"
      profile = imputer.getprofiles()[target]
      assert profile.modelname == modelname

  def test_high_missingness_warning(self, groundtruth: pd.DataFrame) -> None:
    column = "Pre_Semester_GPA"
    subset = groundtruth[[column]].copy()
    subset.loc[subset.index[: int(len(subset) * HIGHMISSINGMASKRATIO)], column] = np.nan
    with warnings.catch_warnings(record = True) as caught :
      warnings.simplefilter("always")
      profilecolumn(column, subset[column])
    assert any(
      str(HIGH_MISSING_THRESHOLD) in str(item.message) or "missing values" in str(item.message).lower()
      for item in caught
    )

  def test_reproducible_output(
    self,
    groundtruth: pd.DataFrame,
    imputablecolumns: list[str],
  ) -> None:
    masked, _ = maskcolumns(groundtruth, columns = imputablecolumns, ratio = MASKRATIO, seed = GUARDSEED, exclude = EXCLUDECOLUMNS)
    first = Simpute(exclude = EXCLUDECOLUMNS, randomstate = GUARDSEED).fit_transform(masked)
    second = Simpute(exclude = EXCLUDECOLUMNS, randomstate = GUARDSEED).fit_transform(masked)
    for column in imputablecolumns :
      if isnumerical(groundtruth[column]) :
        np.testing.assert_allclose(
          first[column].astype(float),
          second[column].astype(float),
          rtol = 1e-10,
          atol = 1e-10,
        )
      else :
        assert first[column].equals(second[column])

  def test_transform_before_fit_raises(self, groundtruth: pd.DataFrame) -> None:
    imputer = Simpute(exclude = EXCLUDECOLUMNS)
    with pytest.raises(RuntimeError, match = "not fitted") :
      imputer.transform(groundtruth)
