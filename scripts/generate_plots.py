"""One-off validation plot generator for README assets. Run from repo root."""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from mask import maskcolumns
from simpute import Simpute
from simpute.utils import isnumerical

DATAPATH = ROOT / "tests" / "data" / "test.csv"
PLOTPATH = ROOT / "Assets" / "Plots"
EXCLUDE = ["Student_ID"]
MASKRATIO = 0.15
SEED = 42
SAMPLE = 5000

COLORS = {
  "orange" : "#F4583E",
  "blue" : "#62C1FE",
  "purple" : "#8957E5",
  "green" : "#2EA043",
  "grey" : "#444C56",
  "gold" : "#D29922",
  "pink" : "#F92672",
}


def styled() -> None:
  plt.rcParams.update({
    "figure.facecolor" : "#22272E",
    "axes.facecolor" : "#22272E",
    "text.color" : "white",
    "axes.labelcolor" : "white",
    "xtick.color" : "white",
    "ytick.color" : "white",
    "axes.edgecolor" : "#22272E",
    "axes.spines.top" : False,
    "axes.spines.right" : False,
    "axes.spines.left" : False,
    "axes.spines.bottom" : False,
    "grid.color" : "#32373E",
  })


def save(fig: plt.Figure, name: str) -> None:
  PLOTPATH.mkdir(parents = True, exist_ok = True)
  fig.savefig(PLOTPATH / name, dpi = 150, facecolor = "#22272E", bbox_inches = "tight")
  plt.close(fig)


def imputation_density(df: pd.DataFrame, result: pd.DataFrame) -> None:
  numerical = [column for column in df.columns if column not in EXCLUDE and isnumerical(df[column])]
  picks = numerical[:3]
  fig, axes = plt.subplots(1, len(picks), figsize = (24, 6))
  if len(picks) == 1 :
    axes = [axes]
  for ax, column in zip(axes, picks) :
    observed = df[column].dropna()
    imputed = result.loc[df[column].isna(), column].dropna()
    sns.kdeplot(observed, ax = ax, color = COLORS["blue"], label = "Observed", linewidth = 2)
    sns.kdeplot(result[column].dropna(), ax = ax, color = COLORS["orange"], label = "Post-imputation", linewidth = 2)
    if len(imputed) :
      sns.kdeplot(imputed, ax = ax, color = COLORS["pink"], label = "Imputed cells", linewidth = 2, linestyle = "--")
    ax.set_title(column, loc = "left", fontsize = 14, pad = 18)
    ax.grid(True, axis = "y", linestyle = "--", alpha = 0.5)
    ax.legend(frameon = False, fontsize = 9)
  fig.suptitle("Imputation Density -- Continuous Variables", x = 0.02, ha = "left", fontsize = 16, y = 1.02)
  fig.tight_layout()
  save(fig, "imputation_density.png")


def missingness_heatmap(df: pd.DataFrame, result: pd.DataFrame) -> None:
  columns = [column for column in df.columns if column not in EXCLUDE]
  before = df[columns].head(SAMPLE).isna().astype(int)
  after = result[columns].head(SAMPLE).isna().astype(int)
  cmap = LinearSegmentedColormap.from_list("missing", [COLORS["green"], COLORS["gold"], COLORS["orange"]])
  fig, axes = plt.subplots(1, 2, figsize = (24, 8), sharey = True)
  for ax, matrix, title in zip(axes, [before, after], ["Before Imputation", "After Imputation"]) :
    sns.heatmap(
      matrix.T,
      ax = ax,
      cmap = cmap,
      cbar = ax is axes[-1],
      yticklabels = columns,
      xticklabels = False,
      vmin = 0,
      vmax = 1,
    )
    ax.set_title(title, loc = "left", fontsize = 14, pad = 18)
  fig.suptitle("Missingness Heatmap", x = 0.02, ha = "left", fontsize = 16, y = 1.02)
  fig.tight_layout()
  save(fig, "missingness_heatmap.png")


def model_allocation_grid(selection: dict[str, str]) -> None:
  counts: dict[str, int] = {}
  for model in selection.values() :
    counts[model] = counts.get(model, 0) + 1
  labels = list(counts.keys())
  values = [counts[label] for label in labels]
  palette = [COLORS["blue"], COLORS["purple"], COLORS["orange"], COLORS["pink"], COLORS["green"], COLORS["gold"]]
  fig, ax = plt.subplots(figsize = (24, 6))
  bars = ax.barh(labels, values, color = palette[: len(labels)])
  ax.set_title("Model Allocation by Column", loc = "left", fontsize = 16, pad = 20)
  ax.grid(True, axis = "x", linestyle = "--", alpha = 0.5)
  for bar, value in zip(bars, values) :
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, str(value), va = "center", fontsize = 11)
  detail = ", ".join(f"{column}={model}" for column, model in sorted(selection.items()))
  ax.set_xlabel(detail, fontsize = 9, labelpad = 12)
  fig.tight_layout()
  save(fig, "model_allocation_grid.png")


def main() -> None:
  styled()
  df = pd.read_csv(DATAPATH)
  columns = [column for column in df.columns if column not in EXCLUDE]
  masked, _ = maskcolumns(df, columns = columns, ratio = MASKRATIO, seed = SEED, exclude = EXCLUDE)
  imputer = Simpute(exclude = EXCLUDE, randomstate = SEED)
  result = imputer.fit_transform(masked)
  imputation_density(df, result)
  missingness_heatmap(masked, result)
  model_allocation_grid(imputer.getmodelselection())
  print(f"Saved plots to {PLOTPATH}")


if __name__ == "__main__" :
  main()
