"""Plot recorded flight evidence: python -m rm27.simulation.plot_micro_flight RUN_DIR."""

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def plot(run_dir):
    run_dir = Path(run_dir).resolve()
    summary = json.loads((run_dir / "summary.json").read_text())
    metadata = json.loads((run_dir / "metadata.json").read_text())
    origin = np.asarray(metadata["origin_nwu"], dtype=float)
    samples, phases = [], []
    with (run_dir / "frames.jsonl").open() as frames:
        for line in frames:
            row = json.loads(line)
            position = np.asarray(row["truth"]["position_nwu"], dtype=float) - origin
            contacts = row["contacts"]
            samples.append(
                [
                    row["state_sent"]["timestamp"],
                    *position,
                    *row["truth"]["actual_thrust_n"],
                    max((contact["force_N"] for contact in contacts), default=0.0),
                    max(
                        (max(0.0, -contact["distance_m"]) for contact in contacts),
                        default=0.0,
                    ),
                ]
            )
            phases.append(row["phase"])
    if not samples:
        raise ValueError("Flight log has no recorded physics frames")
    data = np.asarray(samples, dtype=float)
    if data.shape[1] != 10 or not np.isfinite(data).all():
        raise ValueError("Expected finite position, four motor forces and contact data")
    t = data[:, 0]
    unique_phases = list(dict.fromkeys(phases))
    palette = plt.get_cmap("Set2")
    colors = {name: palette(i % 8) for i, name in enumerate(unique_phases)}
    boundaries = [0] + [i for i in range(1, len(phases)) if phases[i] != phases[i - 1]]
    spans = [
        (phases[start], t[start], t[end] if end < len(t) else t[-1])
        for start, end in zip(boundaries, boundaries[1:] + [len(t)])
    ]
    status = "PASS" if summary["passed"] else "FAIL"
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "savefig.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
        fig.subplots_adjust(top=0.86, bottom=0.11, left=0.09, right=0.9, hspace=0.16)
        fig.suptitle(
            f"{run_dir.name} | {metadata['vehicle']} | recorded flight gate: {status}",
            y=0.975,
            fontsize=16,
        )
        fig.legend(
            handles=[
                Patch(facecolor=colors[name], alpha=0.3, label=name)
                for name in unique_phases
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=5,
            frameon=False,
            fontsize=9,
        )
        for ax in axes:
            for name, start, end in spans:
                ax.axvspan(start, end, color=colors[name], alpha=0.12, linewidth=0)
            ax.set_xlim(t[0], max(t[-1], t[0] + 0.002))

        axes[0].plot(t, data[:, 3], color="#125788", linewidth=1.2)
        axes[0].set_ylabel("Body height above\nslot floor (m)")
        axes[0].axhline(0, color="0.5", linewidth=0.7, linestyle=":")
        axes[1].plot(
            t, data[:, 1], label="World X offset", color="#156086", linewidth=1.1
        )
        axes[1].plot(
            t, data[:, 2], label="World Y offset", color="#b15131", linewidth=1.1
        )
        axes[1].set_ylabel("Position relative to\nassigned slot (m)")
        axes[1].legend(loc="upper left", ncol=2, fontsize=9)
        for i, name in enumerate(("FL", "FR", "RR", "RL")):
            axes[2].plot(t, data[:, 4 + i], linewidth=0.9, label=f"Motor {i}: {name}")
        axes[2].set_ylabel("Actual rotor thrust (N)")
        axes[2].legend(loc="upper left", ncol=4, fontsize=8)
        force_line = axes[3].plot(
            t,
            data[:, 8],
            color="#73428a",
            linewidth=1.0,
            label="Maximum single-contact force",
        )
        axes[3].set_ylabel("Contact force (N)")
        penetration = axes[3].twinx()
        penetration.spines["right"].set_visible(True)
        penetration.grid(False)
        depth_line = penetration.plot(
            t,
            data[:, 9] * 1000,
            color="#c37113",
            linewidth=0.9,
            label="Maximum penetration",
        )
        penetration.set_ylabel("Penetration (mm)")
        axes[3].legend(
            handles=force_line + depth_line, loc="upper left", ncol=2, fontsize=8
        )
        axes[3].set_xlabel("Physics time (s)")
        for event in summary.get("events", []):
            if event["event"] not in ("touchdown", "complete", "failure"):
                continue
            event_time = event["physics_time"]
            if t[0] <= event_time <= t[-1]:
                axes[0].axvline(event_time, color="0.3", linestyle="--", linewidth=0.8)
                axes[0].text(
                    event_time,
                    0.95,
                    event["event"],
                    rotation=90,
                    transform=axes[0].get_xaxis_transform(),
                    va="top",
                    ha="right",
                    fontsize=8,
                )
        footer = (
            f"{len(samples):,} logged physics frames; no resampling. "
            f"Source: {run_dir.name}/frames.jsonl. "
            "Position uses the origin recorded in metadata.json."
        )
        if summary.get("failure"):
            footer += "\n" + textwrap.fill(f"Failure: {summary['failure']}", width=145)
        fig.text(0.09, 0.045, footer, va="center", fontsize=8, color="0.3")
        output = run_dir / "flight.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(plot(args.run_dir))


if __name__ == "__main__":
    main()
