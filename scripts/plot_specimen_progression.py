"""Plot daily mean CAB and MNDVI: uv run python scripts/plot_specimen_progression.py."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from cropspectai import ml_config

INPUT_DIR = ml_config.BASIL_EXPERIMENT_DATA_PATH / "tracked_specimens"
OUTPUT_DIR = ml_config.FIGURES_DIR / "specimen_progression"
METRICS = {"cab": "CAB", "mndvi_as7343": "MNDVI"}


def daily_means(input_dir: Path) -> pd.DataFrame:
    """Combine specimen files and average samples by specimen and calendar date."""
    paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise ValueError(f"No specimen CSV files found in {input_dir}")

    frames = []
    required = ["timestamp", "specimen", *METRICS]
    for path in paths:
        # Detect from the header only: noisy_channels can contain semicolons.
        with path.open(encoding="utf-8-sig", newline="") as handle:
            delimiter = csv.Sniffer().sniff(handle.readline(), delimiters=",;").delimiter
        frame = pd.read_csv(path, sep=delimiter, encoding="utf-8-sig", keep_default_na=False)
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        frame = frame[required].copy()
        if frame.empty:
            continue
        if frame["specimen"].str.strip().eq("").any():
            raise ValueError(f"{path}: blank specimen name")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        if frame["timestamp"].isna().any():
            raise ValueError(f"{path}: missing timestamp")
        for metric in METRICS:
            frame[metric] = pd.to_numeric(frame[metric], errors="raise")
            if not frame[metric].map(math.isfinite).all():
                raise ValueError(f"{path}: missing or non-finite {metric} value")
        # Use the calendar date as recorded, without converting time zones.
        frame["date"] = frame["timestamp"].dt.normalize()
        frames.append(frame)

    if not frames:
        raise ValueError(f"No measurement rows found in {input_dir}")
    return (
        pd.concat(frames, ignore_index=True)
        .groupby(["specimen", "date"], as_index=False, sort=True)
        .agg(
            cab_mean=("cab", "mean"),
            mndvi_mean=("mndvi_as7343", "mean"),
            sample_count=("timestamp", "size"),
        )
    )


def plot_progression(daily: pd.DataFrame):
    """Draw each specimen's daily means in chronological order on shared date axes."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for specimen, values in daily.groupby("specimen", sort=True):
        values = values.sort_values("date")
        for ax, column in zip(axes, ["cab_mean", "mndvi_mean"]):
            ax.plot(values["date"], values[column], marker="o", label=specimen)

    for ax, label in zip(axes, METRICS.values()):
        ax.set_ylabel(f"Mean {label}")
        ax.set_title(f"Daily mean {label}")
        ax.grid(alpha=0.3)
        ax.legend(title="Specimen")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlabel("Measurement date")
    fig.suptitle("Basil specimen progression")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--specimens",
        nargs="+",
        metavar="NAME",
        help="Specimens to include, separated by spaces (default: all specimens)",
    )
    parser.add_argument("--show", action="store_true", help="Also open the plot window")
    args = parser.parse_args()

    daily = daily_means(args.input_dir)
    if args.specimens:
        available = set(daily["specimen"])
        unknown = set(args.specimens) - available
        if unknown:
            parser.error(
                f"Unknown specimens: {', '.join(sorted(unknown))}. "
                f"Available specimens: {', '.join(sorted(available))}"
            )
        daily = daily.loc[daily["specimen"].isin(args.specimens)].copy()
    fig = plot_progression(daily)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "daily_specimen_means.csv"
    plot_path = args.output_dir / "daily_specimen_progression.png"
    daily.to_csv(summary_path, index=False, date_format="%Y-%m-%d")
    fig.savefig(plot_path, dpi=200)
    print(f"Averaged {daily['sample_count'].sum()} samples into {len(daily)} specimen-days.")
    print(f"Saved daily means: {summary_path}")
    print(f"Saved plot: {plot_path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
