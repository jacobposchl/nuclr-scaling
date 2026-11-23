"""Metadata-only comparisons of sampleable time and retained neurons per probe."""

from collections import Counter
import csv
from pathlib import Path

import numpy as np

from src.datasets.allen_sessions import AllenRegistry


DATASETS = (
    ("allen_visual_behavior", "Allen VB", "#3b4cc0"),
    ("allen_visual_coding", "Allen VC", "#b40426"),
    ("ibl_brain_wide_bench_2026", "IBL", "#13866e"),
)


def collect_sampling_metadata(allen_registry, joint_registry, window_seconds=20.):
    """Use full valid domains; preserve source optotagging exclusions and gaps."""
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive and finite")
    allen, joint = AllenRegistry(allen_registry), AllenRegistry(joint_registry)
    recordings, probes = [], []
    for dataset, _, _ in DATASETS:
        registry = joint if dataset.startswith("ibl_") else allen
        spec = registry.spec(dataset)
        if spec["domain"] != "domain" or "stimulus_types" in spec:
            raise ValueError(f"Expected unrestricted recording domains for {dataset}")
        if dataset.startswith("allen_") and "optotagging" not in spec:
            raise ValueError(f"Optotagging exclusion is required for {dataset}")
        sessions = registry.sessions(dataset)
        if not sessions:
            raise ValueError(f"No recordings found for {dataset}")
        for session in sessions:
            description = registry.describe(dataset, session)
            durations = np.array([end - start for start, end in description["intervals"]])
            if not np.isfinite(durations).all() or np.any(durations <= 0):
                raise ValueError(f"Invalid sampling intervals: {dataset}/{session}")
            usable = durations[durations >= window_seconds]
            counts = Counter(row["probe_id"] for row in description["rows"])
            if not len(usable) or not counts:
                raise ValueError(f"No usable windows or retained units: {dataset}/{session}")
            fallback = session in spec.get("optotagging", {}).get("end_at_last_stimulus_sessions", [])
            recordings.append({
                "dataset_id": dataset, "session_id": session,
                "subject_uid": description["subject_uid"],
                "sampleable_minutes": usable.sum() / 60.,
                "allowed_minutes": durations.sum() / 60.,
                "available_windows": int(np.floor(usable / window_seconds).sum()),
                "window_seconds": window_seconds,
                "sampling_intervals": len(durations),
                "neurons": sum(counts.values()), "probes": len(counts),
                "stimulus_end_fallback": fallback,
                "source_file": str(registry.path(dataset, session)),
                "registry_file": str(Path(joint_registry if registry is joint else allen_registry).resolve()),
            })
            probes.extend({"dataset_id": dataset, "session_id": session,
                           "probe_id": probe, "neurons": count}
                          for probe, count in sorted(counts.items()))
        print(f"Read {dataset}: {len(sessions)} recording metadata tables", flush=True)
    return recordings, probes


def save_sampling_distributions(recordings, probes, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    panels = (
        (recordings, "sampleable_minutes", "Available recording duration", "Minutes", 5., "recordings"),
        (probes, "neurons", "Neurons per probe", "Retained neurons per probe", 20., "probes"),
    )
    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 15,
                         "axes.labelsize": 12, "legend.fontsize": 10}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
        for ax, (rows, field, title, xlabel, bin_width, observations) in zip(axes, panels):
            maximum = max(row[field] for row in rows)
            bins = np.arange(0., (np.floor(maximum / bin_width) + 2) * bin_width, bin_width)
            for dataset, label, color in DATASETS:
                values = np.array([row[field] for row in rows if row["dataset_id"] == dataset])
                ax.hist(values, bins=bins, weights=np.full(len(values), 100. / len(values)),
                        histtype="stepfilled", facecolor=color, alpha=.10, edgecolor="none")
                ax.hist(values, bins=bins, weights=np.full(len(values), 100. / len(values)),
                        histtype="step", color=color, linewidth=2.,
                        label=f"{label} ({len(values):,} {observations})")
                quartiles = np.percentile(values, [25, 50, 75])
                summary.append({"dataset_id": dataset, "measurement": field,
                                "count": len(values), "mean": values.mean(),
                                "minimum": values.min(), "q25": quartiles[0],
                                "median": quartiles[1], "q75": quartiles[2],
                                "maximum": values.max()})
            ax.set(title=title, xlabel=xlabel, ylabel=f"Share of {observations} within dataset")
            ax.set_xlim(0, bins[-1])
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.grid(axis="y", color="#d9dfe5", alpha=.6, linewidth=.7)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, loc="upper left" if field == "sampleable_minutes" else "upper right")
        fig.savefig(output_dir / "sampling_distributions.png", dpi=180, facecolor="white")
        plt.close(fig)
    for filename, rows in (("recording_durations.csv", recordings),
                           ("probe_neuron_counts.csv", probes),
                           ("distribution_summary.csv", summary)):
        with (output_dir / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary
