#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Revision analysis for learning-rate sensitivity and statistical significance.

Outputs:
  analysis_revision/
    learning_rate_sensitivity.csv
    learning_rate_sensitivity_paper_table.csv
    statistical_significance_table5.csv
    statistical_significance_backbone.csv
    statistical_significance_label_efficiency.csv
    statistical_significance_five_class.csv
    REVISION_REPORT_FOR_CHATGPT.md
    CHATGPT_UPLOAD_PACKAGE_REVISION.zip
"""
from __future__ import annotations
import os, re, glob, json, zipfile, argparse, math
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    stats = None
    SCIPY_AVAILABLE = False

METRICS = [
    "test_accuracy", "test_macro_f1", "test_weighted_f1",
    "test_moving_accuracy", "test_moving_macro_f1", "test_moving_far"
]

METRIC_LABELS = {
    "test_accuracy": "Accuracy",
    "test_macro_f1": "Macro-F1",
    "test_weighted_f1": "Weighted-F1",
    "test_moving_accuracy": "Moving Accuracy",
    "test_moving_macro_f1": "Moving Macro-F1",
    "test_moving_far": "Moving FAR",
}


def ensure_dir(p: str): os.makedirs(p, exist_ok=True)


def read_summaries(result_root: str) -> pd.DataFrame:
    rows=[]
    for path in glob.glob(os.path.join(result_root, "runs", "*", "fold_*", "seed_*", "run_summary.json")):
        try:
            with open(path, "r", encoding="utf-8") as f: row=json.load(f)
            row["run_dir"] = os.path.dirname(path)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] failed reading {path}: {e}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def parse_lr(exp_id: str, row_lr=None) -> Optional[float]:
    if row_lr is not None and not pd.isna(row_lr):
        try: return float(row_lr)
        except Exception: pass
    # Expected patterns: LR_motion_domain_tcn_1e-4 or LR_motion_domain_tcn_5e-4
    m = re.search(r"_([0-9]+e-[0-9]+)$", str(exp_id))
    if m:
        return float(m.group(1))
    m = re.search(r"_([0-9]+e\+[0-9]+)$", str(exp_id))
    if m:
        return float(m.group(1))
    return None


def fold_seed_aggregate(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    metrics = [m for m in metrics if m in df.columns]
    if df.empty or not metrics: return pd.DataFrame()
    seed_avg = df.groupby(group_cols + ["fold"], dropna=False)[metrics].mean().reset_index()
    rows=[]
    for key, g in seed_avg.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple): key=(key,)
        row={c:v for c,v in zip(group_cols,key)}
        row["num_folds"] = int(g["fold"].nunique())
        for m in metrics:
            vals = g[m].to_numpy(float)
            row[m+"_mean"] = float(np.nanmean(vals))
            row[m+"_std"] = float(np.nanstd(vals, ddof=1)) if len(vals)>1 else 0.0
            row[m+"_ms"] = f"{row[m+'_mean']*100:.2f} ± {row[m+'_std']*100:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def paired_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a=a[mask]; b=b[mask]
    d=a-b
    n=len(d)
    out={"n_pairs": int(n)}
    if n == 0:
        return {**out, "mean_diff": np.nan, "std_diff": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "ttest_p": np.nan, "wilcoxon_p": np.nan, "cohens_dz": np.nan}
    mean=float(np.nanmean(d)); sd=float(np.nanstd(d, ddof=1)) if n>1 else 0.0
    out["mean_diff"] = mean
    out["std_diff"] = sd
    out["mean_diff_pp"] = mean*100
    out["std_diff_pp"] = sd*100
    if n>1 and sd>0:
        if SCIPY_AVAILABLE:
            tcrit = float(stats.t.ppf(0.975, n-1))
        else:
            tcrit = 1.96
        half=tcrit*sd/math.sqrt(n)
        out["ci95_low"] = mean-half
        out["ci95_high"] = mean+half
        out["ci95_low_pp"] = (mean-half)*100
        out["ci95_high_pp"] = (mean+half)*100
        out["cohens_dz"] = mean/sd
    else:
        out["ci95_low"] = mean; out["ci95_high"] = mean
        out["ci95_low_pp"] = mean*100; out["ci95_high_pp"] = mean*100
        out["cohens_dz"] = np.nan
    if SCIPY_AVAILABLE and n>1:
        try: out["ttest_p"] = float(stats.ttest_rel(a,b,nan_policy="omit").pvalue)
        except Exception: out["ttest_p"] = np.nan
        try:
            if np.allclose(d, 0): out["wilcoxon_p"] = 1.0
            else: out["wilcoxon_p"] = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
        except Exception: out["wilcoxon_p"] = np.nan
    else:
        out["ttest_p"] = np.nan; out["wilcoxon_p"] = np.nan
    return out


def compute_comparison(df: pd.DataFrame, name: str, filter_func, proposed_method="motion_domain_proposed") -> pd.DataFrame:
    dfx=df[filter_func(df)].copy()
    if dfx.empty: return pd.DataFrame()
    key_cols=["class_mode","protocol","label_fraction","backbone"]
    rows=[]
    for key,g in dfx.groupby(key_cols, dropna=False):
        base=g[g["fusion_method"]==proposed_method]
        if base.empty: continue
        for method,gm in g.groupby("fusion_method"):
            if method==proposed_method: continue
            merged=base.merge(gm, on=["fold","seed"], suffixes=("_prop","_other"))
            if merged.empty: continue
            for m in [x for x in METRICS if x in df.columns]:
                stats_row=paired_stats(merged[m+"_prop"].to_numpy(), merged[m+"_other"].to_numpy())
                row={"comparison_set": name, "other_method": method, "metric": m, "metric_label": METRIC_LABELS.get(m,m)}
                for c,v in zip(key_cols,key): row[c]=v
                row.update(stats_row)
                rows.append(row)
    return pd.DataFrame(rows)


def make_lr_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lrd = df[df["exp_id"].astype(str).str.startswith("LR_")].copy()
    if lrd.empty:
        return pd.DataFrame(), pd.DataFrame()
    lrd["lr_value"] = [parse_lr(e, r if "lr" in lrd.columns else None) for e,r in zip(lrd["exp_id"], lrd["lr"] if "lr" in lrd.columns else [None]*len(lrd))]
    metrics=[m for m in METRICS if m in lrd.columns]
    agg=fold_seed_aggregate(lrd, ["lr_value","class_mode","protocol","backbone","fusion_method"], metrics)
    if not agg.empty:
        agg=agg.sort_values("lr_value")
    paper_cols=[]
    if not agg.empty:
        for _,r in agg.iterrows():
            paper_cols.append({
                "Learning rate": f"{r['lr_value']:.0e}",
                "Accuracy": r.get("test_accuracy_ms",""),
                "Macro-F1": r.get("test_macro_f1_ms",""),
                "Moving Macro-F1": r.get("test_moving_macro_f1_ms",""),
                "Notes": "selected" if abs(float(r["lr_value"])-5e-4)<1e-12 else ""
            })
    return agg, pd.DataFrame(paper_cols)


def md_table(df: pd.DataFrame, cols: List[str]) -> str:
    if df.empty: return "No data available."
    cols=[c for c in cols if c in df.columns]
    try: return df[cols].to_markdown(index=False)
    except Exception: return df[cols].to_string(index=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--result_root", required=True)
    ap.add_argument("--analysis_dir", default=None)
    args=ap.parse_args()
    out=args.analysis_dir or os.path.join(args.result_root,"analysis_revision")
    ensure_dir(out)
    df=read_summaries(args.result_root)
    if df.empty:
        print("[NO RUNS]"); return
    df.to_csv(os.path.join(out,"all_runs_revision.csv"), index=False)

    lr_agg, lr_paper = make_lr_tables(df)
    lr_agg.to_csv(os.path.join(out,"learning_rate_sensitivity.csv"), index=False)
    lr_paper.to_csv(os.path.join(out,"learning_rate_sensitivity_paper_table.csv"), index=False)

    # Significance sets.
    sigs=[]
    sigs.append(compute_comparison(df, "table5_fusion_tcn", lambda x: x["exp_id"].astype(str).str.startswith("T5_")))
    sigs.append(compute_comparison(df, "backbone_robustness", lambda x: x["exp_id"].astype(str).str.startswith("B_")))
    sigs.append(compute_comparison(df, "label_efficiency", lambda x: x["exp_id"].astype(str).str.startswith("LE_")))
    sigs.append(compute_comparison(df, "five_class", lambda x: x["exp_id"].astype(str).str.startswith("F5_")))
    allsig=pd.concat([s for s in sigs if not s.empty], ignore_index=True) if any(not s.empty for s in sigs) else pd.DataFrame()
    allsig.to_csv(os.path.join(out,"statistical_significance_all.csv"), index=False)
    for cname in ["table5_fusion_tcn", "backbone_robustness", "label_efficiency", "five_class"]:
        sub=allsig[allsig["comparison_set"]==cname] if not allsig.empty else pd.DataFrame()
        sub.to_csv(os.path.join(out,f"statistical_significance_{cname}.csv"), index=False)

    # Create compact paper-ready table: moving macro-F1 and macro-F1 only for table 5.
    if not allsig.empty:
        core=allsig[(allsig["comparison_set"]=="table5_fusion_tcn") & (allsig["metric"].isin(["test_macro_f1","test_moving_macro_f1"]))].copy()
        core.to_csv(os.path.join(out,"table5_core_pvalues.csv"), index=False)

    lines=[]
    lines.append("# Revision Experiment Report for ChatGPT")
    lines.append("")
    lines.append("## Completed run count")
    lines.append(str(len(df)))
    lines.append("")
    lines.append("## Learning-rate sensitivity")
    lines.append(md_table(lr_paper, ["Learning rate","Accuracy","Macro-F1","Moving Macro-F1","Notes"]))
    lines.append("")
    lines.append("## Statistical significance: proposed minus other methods")
    if allsig.empty:
        lines.append("No statistical significance data found. Run Table 5 / backbone / label-efficiency experiments first.")
    else:
        show=allsig[allsig["metric"].isin(["test_macro_f1","test_moving_macro_f1","test_moving_far"])].copy()
        if not show.empty:
            show["diff_pp_ci"] = show.apply(lambda r: f"{r['mean_diff_pp']:.2f} [{r.get('ci95_low_pp',np.nan):.2f}, {r.get('ci95_high_pp',np.nan):.2f}]", axis=1)
            lines.append(md_table(show, ["comparison_set","class_mode","protocol","label_fraction","backbone","other_method","metric_label","n_pairs","diff_pp_ci","ttest_p","wilcoxon_p","cohens_dz"]))
    lines.append("")
    lines.append("## Suggested reviewer response interpretation")
    lines.append("- If proposed vs radar_only is small or not significant, state this explicitly and avoid overclaiming IMU benefit.")
    lines.append("- Emphasize proposed vs early/late/attention fusion and label-scarce non-static-head conditions if those differences are larger.")
    lines.append("- Use the learning-rate table to show whether 5e-4 is stable relative to nearby values.")
    report="\n".join(lines)
    open(os.path.join(out,"REVISION_REPORT_FOR_CHATGPT.md"),"w",encoding="utf-8").write(report)

    summary={
        "result_root": args.result_root,
        "analysis_dir": out,
        "num_runs_total": int(len(df)),
        "num_lr_runs": int(len(df[df['exp_id'].astype(str).str.startswith('LR_')])),
        "scipy_available": SCIPY_AVAILABLE,
    }
    json.dump(summary, open(os.path.join(out,"REVISION_RESULTS_SUMMARY.json"),"w",encoding="utf-8"), indent=2, ensure_ascii=False)

    prompt="""업로드한 REVISION_CHATGPT_UPLOAD_PACKAGE.zip을 바탕으로 major revision 대응용 learning-rate sensitivity와 statistical significance 결과를 분석해줘. 특히 proposed motion-domain protocol이 radar-only 대비 유의한지, early/late/attention fusion 대비 유의한지, learning rate 5e-4가 안정적인지, reviewer response에 어떻게 써야 하는지 알려줘."""
    open(os.path.join(out,"PROMPT_FOR_CHATGPT_REVISION.txt"),"w",encoding="utf-8").write(prompt)

    zip_path=os.path.join(out,"REVISION_CHATGPT_UPLOAD_PACKAGE.zip")
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
        for name in [
            "REVISION_REPORT_FOR_CHATGPT.md", "REVISION_RESULTS_SUMMARY.json", "PROMPT_FOR_CHATGPT_REVISION.txt",
            "all_runs_revision.csv", "learning_rate_sensitivity.csv", "learning_rate_sensitivity_paper_table.csv",
            "statistical_significance_all.csv", "statistical_significance_table5_fusion_tcn.csv", "statistical_significance_backbone_robustness.csv",
            "statistical_significance_label_efficiency.csv", "statistical_significance_five_class.csv", "table5_core_pvalues.csv"
        ]:
            p=os.path.join(out,name)
            if os.path.isfile(p): z.write(p,arcname=name)
    print(f"[REVISION ANALYSIS DONE] {out}")
    print(f"[UPLOAD THIS] {zip_path}")

if __name__ == "__main__":
    main()
