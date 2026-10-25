#!/usr/bin/env python3
"""
scripts/compare_runs.py  ·  Consolidate all evaluation results into a report table.
==================================================================================
Scans outputs/ for metrics*.json files, extracts experiment metadata and
performance scores (Frame-level & Video-level), and compiles them into a unified
comparison table formatted for reports and papers.

Outputs:
    - outputs/comparison_results.csv
    - outputs/comparison_results.md (ready to paste into thesis or GitHub)

Usage:
    python scripts/compare_runs.py
    python scripts/compare_runs.py --outputs-dir outputs --output-file my_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml


def _load_optional_config(run_dir: Path) -> dict:
    """Load config.yaml if present in run directory."""
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


def collect_experiment_metrics(outputs_dir: str | Path = "outputs") -> pd.DataFrame:
    """
    Search outputs/ for all metrics JSON files and aggregate results.
    """
    out_path = Path(outputs_dir)
    if not out_path.exists():
        print(f"[compare_runs] Output directory '{out_path}' does not exist.")
        return pd.DataFrame()

    json_files = sorted(out_path.glob("**/metrics*.json"))
    if not json_files:
        print(f"[compare_runs] No metrics*.json files found under '{out_path}'.")
        return pd.DataFrame()

    records: list[dict] = []

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[compare_runs] Warning: Could not read {jf}: {exc}")
            continue

        run_dir = jf.parent
        cfg_dict = _load_optional_config(run_dir)

        # Extract experiment metadata
        exp_name = data.get("experiment_name", run_dir.name)
        backbone = (
            data.get("backbone")
            or cfg_dict.get("model", {}).get("backbone")
            or "unknown"
        )
        mode = (
            data.get("mode")
            or cfg_dict.get("model", {}).get("mode")
            or cfg_dict.get("model", {}).get("pretraining")
            or "finetune"
        )
        dataset = data.get("test_dataset", "in_domain")

        # Frame-level metrics
        f_m = data.get("frame_metrics", {})
        f_acc = f_m.get("accuracy", 0.0) * 100
        f_auc = f_m.get("auc", 0.5)
        f_f1 = f_m.get("f1", 0.0)
        f_prec = f_m.get("precision", 0.0)
        f_rec = f_m.get("recall", 0.0)

        # Video-level metrics
        v_m = data.get("video_metrics", f_m)
        v_acc = v_m.get("accuracy", 0.0) * 100
        v_auc = v_m.get("auc", 0.5)
        v_f1 = v_m.get("f1", 0.0)
        v_prec = v_m.get("precision", 0.0)
        v_rec = v_m.get("recall", 0.0)

        records.append({
            "Experiment": exp_name,
            "Backbone": backbone,
            "Mode": mode,
            "Test Dataset": dataset,
            "Frame Acc (%)": round(f_acc, 2),
            "Frame AUC": round(f_auc, 4),
            "Frame F1": round(f_f1, 4),
            "Frame Prec": round(f_prec, 4),
            "Frame Rec": round(f_rec, 4),
            "Video Acc (%)": round(v_acc, 2),
            "Video AUC": round(v_auc, 4),
            "Video F1": round(v_f1, 4),
            "Video Prec": round(v_prec, 4),
            "Video Rec": round(v_rec, 4),
            "Run Directory": str(run_dir),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        # Sort by Video AUC descending
        df = df.sort_values(by=["Video AUC", "Frame AUC"], ascending=False).reset_index(drop=True)
    return df


def generate_comparison_reports(
    outputs_dir: str | Path = "outputs",
    save_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Collect metrics and export CSV and Markdown comparison tables.
    """
    outputs_dir = Path(outputs_dir)
    save_dir = Path(save_dir) if save_dir is not None else outputs_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    df = collect_experiment_metrics(outputs_dir)
    if df.empty:
        print("[compare_runs] No experiment results to compile.")
        return df

    csv_path = save_dir / "comparison_results.csv"
    md_path = save_dir / "comparison_results.md"

    # Save full CSV
    df.to_csv(csv_path, index=False)

    # Simplified display columns for markdown table
    display_cols = [
        "Experiment",
        "Backbone",
        "Mode",
        "Test Dataset",
        "Frame Acc (%)",
        "Frame AUC",
        "Frame F1",
        "Video Acc (%)",
        "Video AUC",
        "Video F1",
    ]
    display_df = df[[c for c in display_cols if c in df.columns]]
    markdown_table = display_df.to_markdown(index=False)

    md_content = (
        f"# 📊 Deepfake Detection: Comparative Experiment Results\n\n"
        f"Generated automatically from evaluation metrics in `{outputs_dir}`.\n\n"
        f"{markdown_table}\n\n"
        f"---\n"
        f"*Note: Video-level metrics are computed by aggregating per-frame probabilities.*"
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 80)
    print("  [BENCHMARK] Comparison Results Table")
    print("=" * 80)
    print(markdown_table)
    print("=" * 80)
    print(f"\nSaved CSV report to      : {csv_path.resolve()}")
    print(f"Saved Markdown report to : {md_path.resolve()}\n")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and compare experiment results")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Directory containing experiment runs.")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save summary CSV and Markdown.")
    args = parser.parse_args()

    generate_comparison_reports(outputs_dir=args.outputs_dir, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
