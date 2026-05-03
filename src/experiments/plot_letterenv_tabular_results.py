"""Generate thesis-style figures for tabular LetterEnv encoding comparisons."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

matplotlib_cache_dir = Path(tempfile.gettempdir()) / "final-project-matplotlib"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.legacy_paths import code_root


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ENCODINGS = ("simple", "one_hot", "numerical")


@dataclass(frozen=True)
class RunSelection:
    """Resolved input directory for one encoding."""

    encoding: str
    run_dir: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a thesis-style comparison figure for the tabular LetterEnv "
            "baseline across monitor-state encodings."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=code_root() / "src" / "results" / "tabular_letterenv",
        help="Root directory containing encoding-specific run directories.",
    )
    parser.add_argument(
        "--encodings",
        nargs="+",
        choices=ENCODINGS,
        default=list(ENCODINGS),
        help="Encodings to include in the figure.",
    )
    parser.add_argument(
        "--simple-run",
        type=Path,
        default=None,
        help="Explicit run directory for the simple encoding.",
    )
    parser.add_argument(
        "--one-hot-run",
        type=Path,
        default=None,
        help="Explicit run directory for the one-hot encoding.",
    )
    parser.add_argument(
        "--numerical-run",
        type=Path,
        default=None,
        help="Explicit run directory for the numerical encoding.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory in which to save the figure and summary table. If "
            "omitted, the figure is saved under results_root/figures/latest."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="LetterEnv Tabular Baseline Comparison",
        help="Figure title.",
    )
    return parser.parse_args()


def resolve_output_dir(results_root: Path, requested: Path | None) -> Path:
    """Resolve the output directory for generated artifacts."""
    if requested is None:
        return results_root / "figures" / "latest"
    return requested if requested.is_absolute() else code_root() / requested


def latest_run_dir(results_root: Path, encoding: str) -> Path:
    """Return the most recent run directory for an encoding."""
    encoding_root = results_root / encoding
    if not encoding_root.exists():
        raise FileNotFoundError(
            f"No results directory found for encoding '{encoding}' at {encoding_root}."
        )

    candidates = sorted(
        (
            path
            for path in encoding_root.iterdir()
            if path.is_dir() and (path / "train_metrics.csv").exists()
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No completed run directories containing train_metrics.csv were found under "
            f"{encoding_root}."
        )
    return candidates[-1]


def resolve_run_selections(args: argparse.Namespace) -> list[RunSelection]:
    """Resolve the run directory used for each selected encoding."""
    explicit_paths = {
        "simple": args.simple_run,
        "one_hot": args.one_hot_run,
        "numerical": args.numerical_run,
    }

    selections: list[RunSelection] = []
    for encoding in args.encodings:
        requested = explicit_paths[encoding]
        if requested is None:
            run_dir = latest_run_dir(args.results_root, encoding)
        else:
            run_dir = requested if requested.is_absolute() else code_root() / requested
        if not (run_dir / "train_metrics.csv").exists():
            raise FileNotFoundError(
                f"Expected train_metrics.csv in {run_dir}, but the file was not found."
            )
        selections.append(RunSelection(encoding=encoding, run_dir=run_dir))

    return selections


def load_run_frame(selection: RunSelection) -> pd.DataFrame:
    """Load one run directory into a standardized DataFrame."""
    frame = pd.read_csv(selection.run_dir / "train_metrics.csv")
    frame["encoding"] = selection.encoding
    frame["run_dir"] = str(selection.run_dir)
    return frame


def load_all_runs(selections: list[RunSelection]) -> pd.DataFrame:
    """Load and concatenate all selected runs."""
    frames = [load_run_frame(selection) for selection in selections]
    return pd.concat(frames, ignore_index=True)


def summarize_for_plot(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean and standard deviation by encoding and n value."""
    summary = (
        results.groupby(["encoding", "n_value"], sort=True)
        .agg(
            mean_steps=("steps", "mean"),
            std_steps=("steps", lambda values: float(np.std(values, ddof=0))),
            mean_episodes=("episodes", "mean"),
            std_episodes=("episodes", lambda values: float(np.std(values, ddof=0))),
            converged_fraction=("converged", "mean"),
            count=("n_value", "size"),
        )
        .reset_index()
    )
    return summary


def build_summary_payload(
    selections: list[RunSelection],
    summary: pd.DataFrame,
    encodings: list[str],
) -> dict[str, Any]:
    """Build JSON metadata for the generated figure."""
    payload: dict[str, Any] = {
        "runs": {selection.encoding: str(selection.run_dir) for selection in selections},
        "encodings": {},
    }
    for encoding in encodings:
        encoding_rows = summary.loc[summary["encoding"] == encoding]
        payload["encodings"][encoding] = encoding_rows.to_dict(orient="records")
    return payload


def plot_metric(
    axis: plt.Axes,
    summary: pd.DataFrame,
    encodings: list[str],
    metric_mean: str,
    metric_std: str,
    ylabel: str,
) -> None:
    """Plot one thesis-style line chart with standard deviation bands."""
    palette = {
        "simple": "#c58b00",
        "one_hot": "#1f77b4",
        "numerical": "#d62728",
    }
    markers = {
        "simple": "o",
        "one_hot": "s",
        "numerical": "^",
    }
    labels = {
        "simple": "Simple",
        "one_hot": "One-Hot",
        "numerical": "Numerical",
    }

    for encoding in encodings:
        rows = summary.loc[summary["encoding"] == encoding].sort_values("n_value")
        x_values = rows["n_value"].to_numpy(dtype=float)
        mean_values = rows[metric_mean].to_numpy(dtype=float)
        std_values = rows[metric_std].to_numpy(dtype=float)

        axis.plot(
            x_values,
            mean_values,
            label=labels[encoding],
            color=palette[encoding],
            marker=markers[encoding],
            linewidth=2.0,
            markersize=6.0,
        )
        axis.fill_between(
            x_values,
            mean_values - std_values,
            mean_values + std_values,
            color=palette[encoding],
            alpha=0.18,
        )

    axis.set_xlabel("Task parameter n")
    axis.set_ylabel(ylabel)
    axis.set_xticks(sorted(summary["n_value"].unique()))


def create_figure(summary: pd.DataFrame, encodings: list[str], title: str) -> plt.Figure:
    """Create the comparison figure."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    plot_metric(axes[0], summary, encodings, "mean_steps", "std_steps", "Steps to convergence")
    plot_metric(
        axes[1],
        summary,
        encodings,
        "mean_episodes",
        "std_episodes",
        "Episodes to convergence",
    )

    axes[0].set_title("Training Steps")
    axes[1].set_title("Training Episodes")
    axes[1].legend(loc="upper left")
    figure.suptitle(title)

    return figure


def save_outputs(
    *,
    output_dir: Path,
    figure: plt.Figure,
    summary: pd.DataFrame,
    payload: dict[str, Any],
) -> None:
    """Save the figure and its supporting tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "letterenv_tabular_comparison.png"
    pdf_path = output_dir / "letterenv_tabular_comparison.pdf"
    csv_path = output_dir / "letterenv_tabular_summary.csv"
    json_path = output_dir / "letterenv_tabular_summary.json"

    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    summary.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """Entrypoint for figure generation."""
    args = parse_args()
    selections = resolve_run_selections(args)
    results = load_all_runs(selections)
    summary = summarize_for_plot(results)
    figure = create_figure(summary, args.encodings, args.title)
    payload = build_summary_payload(selections, summary, args.encodings)
    output_dir = resolve_output_dir(args.results_root, args.output_dir)
    save_outputs(output_dir=output_dir, figure=figure, summary=summary, payload=payload)
    plt.close(figure)

    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "runs": {selection.encoding: str(selection.run_dir) for selection in selections},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
