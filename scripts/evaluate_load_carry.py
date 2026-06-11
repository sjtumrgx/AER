"""Aggregate and plot load-carry evaluation metrics.

This script is intentionally lightweight: it can either write a required-plot
manifest before rollouts exist, or aggregate a CSV produced by play/evaluation
rollouts. Expected CSV columns include `condition`, `step`, and any metric key in
`gym_learn.eval_metrics.metrics.LOAD_CARRY_PLOT_SPECS`.
"""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from gym_learn.eval_metrics.metrics import LOAD_CARRY_PLOT_SPECS


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--metrics_csv", default=None, type=str, help="CSV with condition, step, and load-carry metric columns.")
    parser.add_argument("--output_dir", default="analysis/load_carry", type=str, help="Directory for summary CSV, manifest, and plots.")
    parser.add_argument("--payload_masses", default="0,2,4,6,8", type=str, help="Comma-separated payload masses for manifest conditions.")
    parser.add_argument("--com_offsets", default="0.0,0.04,-0.04", type=str, help="Comma-separated CoM x-offsets for manifest conditions.")
    parser.add_argument("--dynamic_payload", action="store_true", help="Include dynamic payload condition labels in the manifest.")
    parser.add_argument("--dry_run", action="store_true", help="Write manifest only, without requiring metrics_csv.")
    return parser


def _parse_floats(csv_text: str) -> List[float]:
    return [float(item) for item in csv_text.split(",") if item.strip()]


def build_conditions(payload_masses: Iterable[float], com_offsets: Iterable[float], dynamic_payload: bool) -> List[Dict[str, object]]:
    conditions = []
    for mass in payload_masses:
        for offset in com_offsets:
            conditions.append({"name": f"mass_{mass:g}_comx_{offset:g}_static", "payload_mass": mass, "com_offset_x": offset, "dynamic_payload": False})
            if dynamic_payload:
                conditions.append({"name": f"mass_{mass:g}_comx_{offset:g}_dynamic", "payload_mass": mass, "com_offset_x": offset, "dynamic_payload": True})
    return conditions


def write_manifest(output_dir: Path, conditions: List[Dict[str, object]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {name: str(output_dir / f"{name}.png") for name in LOAD_CARRY_PLOT_SPECS}
    manifest = {
        "conditions": conditions,
        "plots": plot_paths,
        "required_metrics": list(LOAD_CARRY_PLOT_SPECS.keys()),
    }
    path = output_dir / "load_carry_plot_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return path


def read_metric_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_summary(rows: List[Dict[str, str]], output_dir: Path) -> Path:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("condition", "default"), []).append(row)

    metric_names = [name for name in LOAD_CARRY_PLOT_SPECS if any(name in row and row[name] != "" for row in rows)]
    summary_path = output_dir / "load_carry_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", *metric_names])
        writer.writeheader()
        for condition, condition_rows in sorted(grouped.items()):
            out = {"condition": condition}
            for metric in metric_names:
                values = [float(row[metric]) for row in condition_rows if row.get(metric) not in (None, "")]
                out[metric] = mean(values) if values else ""
            writer.writerow(out)
    return summary_path


def maybe_plot(rows: List[Dict[str, str]], output_dir: Path) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written = []
    for metric in LOAD_CARRY_PLOT_SPECS:
        if not any(metric in row and row[metric] != "" for row in rows):
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        by_condition: Dict[str, List[tuple]] = {}
        for row in rows:
            if row.get(metric) in (None, ""):
                continue
            step = float(row.get("step") or len(by_condition.get(row.get("condition", "default"), [])))
            by_condition.setdefault(row.get("condition", "default"), []).append((step, float(row[metric])))
        for condition, points in sorted(by_condition.items()):
            points.sort()
            ax.plot([p[0] for p in points], [p[1] for p in points], label=condition)
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.set_ylabel(LOAD_CARRY_PLOT_SPECS[metric]["ylabel"])
        if len(by_condition) <= 12:
            ax.legend(fontsize="small")
        fig.tight_layout()
        out = output_dir / f"{metric}.png"
        fig.savefig(out)
        plt.close(fig)
        written.append(out)
    return written


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    conditions = build_conditions(_parse_floats(args.payload_masses), _parse_floats(args.com_offsets), args.dynamic_payload)
    manifest_path = write_manifest(output_dir, conditions)

    if args.dry_run or args.metrics_csv is None:
        print(f"Wrote load-carry plot manifest: {manifest_path}")
        return 0

    rows = read_metric_rows(Path(args.metrics_csv))
    summary_path = write_summary(rows, output_dir)
    plot_paths = maybe_plot(rows, output_dir)
    print(f"Wrote summary: {summary_path}")
    if plot_paths:
        print("Wrote plots:")
        for path in plot_paths:
            print(f"  {path}")
    else:
        print("No plots written (matplotlib unavailable or CSV lacks load-carry metric columns).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
