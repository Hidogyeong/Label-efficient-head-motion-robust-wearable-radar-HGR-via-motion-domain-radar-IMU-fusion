#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate fusion-protocol experiment results and package them for ChatGPT review."""
from __future__ import annotations

import os
import json
import glob
import zipfile
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_summaries(result_root: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(result_root, "runs", "*", "fold_*", "seed_*", "run_summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["run_dir"] = os.path.dirname(path)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def mean_std_str(mean: float, std: float, pct: bool = True) -> str:
    if np.isnan(mean):
        return ""
    if pct:
        return f"{mean*100:.2f} ± {std*100:.2f}"
    return f"{mean:.4f} ± {std:.4f}"


def aggregate_fold_seed(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    # First average over seeds within each fold, then aggregate folds.
    seed_avg = df.groupby(group_cols + ["fold"], dropna=False)[metric_cols].mean().reset_index()
    rows = []
    for key, g in seed_avg.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        row["num_folds"] = int(g["fold"].nunique())
        for m in metric_cols:
            row[m + "_mean"] = float(g[m].mean())
            row[m + "_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
            row[m + "_ms"] = mean_std_str(row[m + "_mean"], row[m + "_std"], pct=True)
        rows.append(row)
    return pd.DataFrame(rows)


def read_head_class_metrics(result_root: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    head_rows = []
    class_rows = []
    for _, r in df.iterrows():
        run_dir = r["run_dir"]
        hp = os.path.join(run_dir, "test_metrics_by_head.csv")
        cp = os.path.join(run_dir, "test_metrics_by_class.csv")
        meta_cols = ["exp_id", "fold", "seed", "class_mode", "protocol", "label_fraction", "fusion_method", "backbone"]
        meta = {c: r[c] for c in meta_cols if c in r}
        if os.path.isfile(hp):
            hdf = pd.read_csv(hp)
            for k, v in meta.items():
                hdf[k] = v
            head_rows.append(hdf)
        if os.path.isfile(cp):
            cdf = pd.read_csv(cp)
            for k, v in meta.items():
                cdf[k] = v
            class_rows.append(cdf)
    return (pd.concat(head_rows, ignore_index=True) if head_rows else pd.DataFrame(),
            pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame())


def paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Compute paired differences against motion_domain_proposed for each backbone/protocol/class."""
    metric_cols = ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_moving_accuracy", "test_moving_macro_f1", "test_moving_far"]
    rows = []
    keys = ["class_mode", "protocol", "backbone"]
    for key, g in df.groupby(keys, dropna=False):
        base = g[g["fusion_method"] == "motion_domain_proposed"]
        if base.empty:
            continue
        for method, gm in g.groupby("fusion_method"):
            if method == "motion_domain_proposed":
                continue
            merged = base.merge(gm, on=["fold", "seed"], suffixes=("_prop", "_other"))
            if merged.empty:
                continue
            row = {col: val for col, val in zip(keys, key)}
            row["other_method"] = method
            row["num_pairs"] = int(len(merged))
            for m in metric_cols:
                a = merged[m + "_prop"].to_numpy(dtype=float)
                b = merged[m + "_other"].to_numpy(dtype=float)
                d = a - b
                row[m + "_diff_mean"] = float(np.nanmean(d))
                row[m + "_diff_std"] = float(np.nanstd(d, ddof=1)) if len(d) > 1 else 0.0
                row[m + "_diff_pp"] = float(np.nanmean(d) * 100)
            rows.append(row)
    return pd.DataFrame(rows)


def make_report(analysis_dir: str, all_runs: pd.DataFrame, fusion_table: pd.DataFrame, label_eff: pd.DataFrame, paired: pd.DataFrame) -> str:
    lines = []
    lines.append("# ChatGPT Handoff Report: Radar-IMU Fusion Protocol Experiments")
    lines.append("")
    lines.append("## 1. What this experiment package tests")
    lines.append("This package was created after supervisor feedback that the previous Table 5 compared only network backbones. The new experiments compare fusion/training strategies under the same wearable radar-IMU subject-disjoint protocol.")
    lines.append("")
    lines.append("Fusion/training strategies:")
    lines.append("- radar_only: radar time-series only")
    lines.append("- early_fusion: radar + IMU concatenated at each time step")
    lines.append("- late_fusion: radar and IMU encoded separately then concatenated")
    lines.append("- attention_fusion: IMU-guided cross-attention over radar features")
    lines.append("- motion_domain_proposed: Doppler-aware motion-XYZ representation")
    lines.append("")
    lines.append("## 2. Completed runs")
    if all_runs.empty:
        lines.append("No runs found.")
    else:
        lines.append(f"Total completed runs: {len(all_runs)}")
        pivot = all_runs.groupby(["exp_id", "fusion_method", "backbone", "class_mode", "protocol"]).size().reset_index(name="runs")
        lines.append("")
        lines.append(pivot.to_markdown(index=False))
    lines.append("")
    lines.append("## 3. Aggregated fusion/protocol results")
    if fusion_table.empty:
        lines.append("No fusion table available.")
    else:
        show_cols = [c for c in ["class_mode","protocol","label_fraction","backbone","fusion_method","num_folds","test_accuracy_ms","test_macro_f1_ms","test_weighted_f1_ms","test_moving_accuracy_ms","test_moving_macro_f1_ms","test_moving_far_ms"] if c in fusion_table.columns]
        lines.append(fusion_table[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 4. Label-efficiency results")
    if label_eff.empty:
        lines.append("No label-efficiency table available.")
    else:
        show_cols = [c for c in ["class_mode","backbone","fusion_method","label_fraction","num_folds","test_macro_f1_ms","test_moving_macro_f1_ms","test_moving_far_ms"] if c in label_eff.columns]
        lines.append(label_eff[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 5. Paired comparisons: proposed minus existing fusion method")
    if paired.empty:
        lines.append("No paired comparison available.")
    else:
        show_cols = [c for c in ["class_mode","protocol","backbone","other_method","num_pairs","test_macro_f1_diff_pp","test_moving_macro_f1_diff_pp","test_moving_far_diff_pp"] if c in paired.columns]
        lines.append(paired[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 6. How to interpret")
    lines.append("If motion_domain_proposed is better than radar_only, early_fusion, late_fusion, and attention_fusion under the same backbone, the paper can claim that physics-guided motion-domain representation is more effective than raw data-driven sensor fusion for this wearable radar-IMU HGR setting.")
    lines.append("If early_fusion or attention_fusion is better, the paper direction should be revised toward an empirical benchmark rather than a proposed protocol paper.")
    lines.append("")
    lines.append("## 7. Files to upload together")
    lines.append("- all_runs.csv")
    lines.append("- fusion_strategy_aggregate.csv")
    lines.append("- label_efficiency_aggregate.csv")
    lines.append("- paired_comparisons.csv")
    lines.append("- head_metrics_aggregate.csv")
    lines.append("- class_metrics_aggregate.csv")
    lines.append("")
    report = "\n".join(lines)
    with open(os.path.join(analysis_dir, "REPORT_FOR_CHATGPT.md"), "w", encoding="utf-8") as f:
        f.write(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=str, required=True)
    parser.add_argument("--analysis_dir", type=str, default=None)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir or os.path.join(args.result_root, "analysis_fusion_protocol")
    ensure_dir(analysis_dir)

    all_runs = read_summaries(args.result_root)
    if all_runs.empty:
        print("[NO RUNS] No run_summary.json files found.")
        return
    all_runs.to_csv(os.path.join(analysis_dir, "all_runs.csv"), index=False)

    metric_cols = [c for c in [
        "test_accuracy", "test_macro_f1", "test_weighted_f1", "test_moving_accuracy",
        "test_moving_macro_f1", "test_moving_weighted_f1", "test_far", "test_moving_far",
        "best_val_macro_f1", "num_params"
    ] if c in all_runs.columns]
    group_cols = ["class_mode", "protocol", "label_fraction", "backbone", "fusion_method"]
    fusion_table = aggregate_fold_seed(all_runs, group_cols, metric_cols)
    fusion_table.to_csv(os.path.join(analysis_dir, "fusion_strategy_aggregate.csv"), index=False)

    label_eff = fusion_table[fusion_table["protocol"].astype(str).eq("partial_mixed")].copy()
    label_eff.to_csv(os.path.join(analysis_dir, "label_efficiency_aggregate.csv"), index=False)

    paired = paired_comparisons(all_runs)
    paired.to_csv(os.path.join(analysis_dir, "paired_comparisons.csv"), index=False)

    head_all, class_all = read_head_class_metrics(args.result_root, all_runs)
    if not head_all.empty:
        head_all.to_csv(os.path.join(analysis_dir, "head_metrics_all_runs.csv"), index=False)
        head_agg = aggregate_fold_seed(head_all, ["class_mode", "protocol", "label_fraction", "backbone", "fusion_method", "head"], ["accuracy", "macro_f1", "weighted_f1"] + (["far"] if "far" in head_all.columns else []))
        head_agg.to_csv(os.path.join(analysis_dir, "head_metrics_aggregate.csv"), index=False)
    else:
        head_agg = pd.DataFrame()
    if not class_all.empty:
        class_all.to_csv(os.path.join(analysis_dir, "class_metrics_all_runs.csv"), index=False)
        class_agg = aggregate_fold_seed(class_all, ["class_mode", "protocol", "label_fraction", "backbone", "fusion_method", "class_name"], ["precision", "recall", "f1"])
        class_agg.to_csv(os.path.join(analysis_dir, "class_metrics_aggregate.csv"), index=False)
    else:
        class_agg = pd.DataFrame()

    summary = {
        "result_root": args.result_root,
        "analysis_dir": analysis_dir,
        "num_runs": int(len(all_runs)),
        "experiments": sorted(all_runs["exp_id"].unique().tolist()),
        "fusion_methods": sorted(all_runs["fusion_method"].unique().tolist()),
        "backbones": sorted(all_runs["backbone"].unique().tolist()),
    }
    with open(os.path.join(analysis_dir, "CHATGPT_RESULTS_SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    make_report(analysis_dir, all_runs, fusion_table, label_eff, paired)

    prompt = """업로드한 CHATGPT_UPLOAD_PACKAGE.zip을 바탕으로 fusion/training protocol 실험 결과를 분석해줘. 특히 motion_domain_proposed가 radar_only, early_fusion, late_fusion, attention_fusion보다 좋은지, CNN/TCN/GRU/Transformer backbone에서 그 경향이 유지되는지, 논문 방향을 유지할 수 있는지 판단해줘."""
    with open(os.path.join(analysis_dir, "PROMPT_FOR_CHATGPT.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)

    zip_path = os.path.join(analysis_dir, "CHATGPT_UPLOAD_PACKAGE.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in [
            "REPORT_FOR_CHATGPT.md", "CHATGPT_RESULTS_SUMMARY.json", "PROMPT_FOR_CHATGPT.txt",
            "all_runs.csv", "fusion_strategy_aggregate.csv", "label_efficiency_aggregate.csv",
            "paired_comparisons.csv", "head_metrics_aggregate.csv", "class_metrics_aggregate.csv",
            "head_metrics_all_runs.csv", "class_metrics_all_runs.csv",
        ]:
            p = os.path.join(analysis_dir, name)
            if os.path.isfile(p):
                z.write(p, arcname=name)
    print(f"[ANALYSIS DONE] {analysis_dir}")
    print(f"[UPLOAD THIS] {zip_path}")

if __name__ == "__main__":
    main()
