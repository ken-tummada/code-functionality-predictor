import json
import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")  # remove for interactive display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


# ──────────────────────────────────────────────────────────────────────────────
# 📥 LOAD YOUR DATA
#
# Each JSON file represents ONE judge model scoring ALL inference models.
# Structure per file:
#   {
#     "sonnet-4.5": {
#       "intention":           [4, 4, 3, ...],
#       "technical_specs":     [4, 3, 4, ...],
#       "abstraction_level":   [3, 3, 2, ...],
#       "implementation_ease": [3, 4, 3, ...]
#     },
#     "gpt-5-mini": { ... },
#     ...
#   }
#
# Replace the paths below with your actual files:
# ──────────────────────────────────────────────────────────────────────────────

JUDGE_FILES = {
    "sonnet-4.5": "results/desc-gen-sonnet-4.5/raw.json",
    "gpt-5-mini": "results/desc-gen-llama-3-8b/raw.json",
    "llama-3-8b": "results/desc-gen-gpt-5-mini/raw.json",
}


def load_judge_data(judge_files: dict) -> dict:
    """Load all judge JSON files into a nested dict.

    Returns
    -------
    dict : { judge_model: { inference_model: { dimension: [scores] } } }
    """
    data = {}
    for judge, path in judge_files.items():
        with open(path, "r", encoding="utf-8") as f:
            data[judge] = json.load(f)
    return data


# ──────────────────────────────────────────────────────────────────────────────
# 🔄 OPTIONAL: build from Polars DataFrames instead of JSON files
#
# If you have one pl.DataFrame per judge (columns: model, dimension, score_list):
#
# def polars_to_judge_dict(df: pl.DataFrame, judge_name: str) -> dict:
#     result = {}
#     for row in df.iter_rows(named=True):
#         result.setdefault(row["model"], {})[row["dimension"]] = row["scores"]
#     return result
# ──────────────────────────────────────────────────────────────────────────────


def build_matrix(judge_data: dict, models: list, judges: list, dim: str) -> np.ndarray:
    """
    Build a (n_models × n_judges) matrix for one dimension.

    cell[i, j] = mean score that judge j gave to inference model i
                 on the given dimension.
    """
    mat = np.zeros((len(models), len(judges)))
    for j, judge in enumerate(judges):
        for i, inf_model in enumerate(models):
            scores = judge_data[judge][inf_model][dim]
            mat[i, j] = round(np.mean(scores), 2)
    return mat


def plot_pairwise_matrices(judge_data: dict, output_file: str = None) -> None:
    """
    Plot a 2×2 grid of pairwise evaluation heatmaps — one per dimension.

    Rows   = inference model (the model being evaluated)
    Cols   = judge model     (the model doing the scoring)
    Cell   = mean score the judge gave to the inference model

    Parameters
    ----------
    judge_data : dict
        Nested dict: { judge: { inference_model: { dimension: [scores] } } }
    output_file : str, optional
        Save path (e.g. "output.png"). None = display interactively.
    """
    judges = list(judge_data.keys())
    models = list(next(iter(judge_data.values())).keys())
    dimensions = list(next(iter(next(iter(judge_data.values())).values())).keys())

    matrices = {
        dim: build_matrix(judge_data, models, judges, dim) for dim in dimensions
    }

    # Shared colour scale across all 4 subplots for visual consistency
    all_vals = np.concatenate([m.flatten() for m in matrices.values()])
    vmin, vmax = all_vals.min(), all_vals.max()

    dim_labels = {d: d.replace("_", " ").title() for d in dimensions}

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(
        "Pairwise Model Evaluation\n"
        "rows = inference model  ·  cols = judge model  ·  cell = mean score",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4)

    for idx, dim in enumerate(dimensions):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        mat = matrices[dim]

        hm = sns.heatmap(
            mat,
            ax=ax,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            linewidths=1.2,
            linecolor="white",
            square=True,
            cbar_kws={"shrink": 0.8, "label": "mean score"},
            xticklabels=judges,
            yticklabels=models,
            annot_kws={"size": 11},
        )
        hm.collections[0].colorbar.ax.tick_params(labelsize=8)

        ax.set_title(dim_labels[dim], fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Judge →", fontsize=9, labelpad=6)
        ax.set_ylabel("← Inference", fontsize=9, labelpad=6)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_file}")
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    judge_data = load_judge_data(JUDGE_FILES)
    plot_pairwise_matrices(judge_data, output_file="pairwise_matrix_heatmaps.png")
