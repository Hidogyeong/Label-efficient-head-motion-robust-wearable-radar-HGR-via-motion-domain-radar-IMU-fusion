#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import argparse
import pandas as pd

GESTURE_ORDER_5 = ["01RtoL", "02LtoR", "03UtoD", "04DtoU", "05None"]
GESTURE_ORDER_4 = ["01RtoL", "02LtoR", "03UtoD", "04DtoU"]

HEAD_ORDER = ["None", "Down", "Left", "Random", "Right", "UP"]

# 논문 표에서는 UP보다 Up 표기가 자연스러워서 출력용 이름만 바꿈
HEAD_PRINT = {
    "None": "None",
    "Down": "Down",
    "Left": "Left",
    "Random": "Random",
    "Right": "Right",
    "UP": "Up",
    "Up": "Up",
}

def normalize_head_name(h):
    if h == "Up":
        return "UP"
    return h

def scan_dataset(data_root: Path):
    rows = []

    for person_dir in sorted(data_root.iterdir()):
        if not person_dir.is_dir():
            continue

        person = person_dir.name

        for gesture_dir in sorted(person_dir.iterdir()):
            if not gesture_dir.is_dir():
                continue

            gesture = gesture_dir.name

            for head_dir in sorted(gesture_dir.iterdir()):
                if not head_dir.is_dir():
                    continue

                head = normalize_head_name(head_dir.name)

                csv_files = list(head_dir.rglob("*.csv"))

                for f in csv_files:
                    rows.append({
                        "person": person,
                        "gesture": gesture,
                        "head": head,
                        "file_path": str(f),
                    })

    return pd.DataFrame(rows)

def make_distribution(df, gestures):
    sub = df[df["gesture"].isin(gestures)].copy()

    table = (
        sub
        .pivot_table(
            index="gesture",
            columns="head",
            values="file_path",
            aggfunc="count",
            fill_value=0
        )
    )

    for h in HEAD_ORDER:
        if h not in table.columns:
            table[h] = 0

    table = table[HEAD_ORDER]
    table = table.reindex(gestures, fill_value=0)
    table["Total"] = table.sum(axis=1)

    total_row = pd.DataFrame(table.sum(axis=0)).T
    total_row.index = ["Total"]

    table = pd.concat([table, total_row], axis=0)

    # 출력용 column 이름 변경
    table = table.rename(columns={h: HEAD_PRINT.get(h, h) for h in table.columns})

    return table.astype(int)

def latex_table(table, caption, label):
    # LaTeX에서 gesture 이름을 보기 좋게 표시
    display_index = {
        "01RtoL": "01RtoL",
        "02LtoR": "02LtoR",
        "03UtoD": "03UtoD",
        "04DtoU": "04DtoU",
        "05None": "05None",
        "Total": "Total",
    }

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Gesture & None & Down & Left & Random & Right & Up & Total \\")
    lines.append(r"\midrule")

    for idx, row in table.iterrows():
        name = display_index.get(idx, idx)
        vals = [int(row[c]) for c in ["None", "Down", "Left", "Random", "Right", "Up", "Total"]]
        if idx == "Total":
            lines.append(r"\midrule")
            lines.append(
                rf"\textbf{{{name}}} & "
                + " & ".join([rf"\textbf{{{v}}}" for v in vals])
                + r" \\"
            )
        else:
            lines.append(
                f"{name} & "
                + " & ".join(str(v) for v in vals)
                + r" \\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/media/dokyeong/MINJI/dokyeong/processedData_radar_imu_geometry_GPT_2",
        help="Dataset root path: person/gesture/head/*.csv"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/media/dokyeong/MINJI/dokyeong/RadarIMU_HGR_FusionProtocol_Experiments/revision_tables",
        help="Output directory"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    df = scan_dataset(data_root)

    if len(df) == 0:
        raise RuntimeError(f"No CSV files found under {data_root}")

    # raw file list
    df.to_csv(out_dir / "all_samples_file_list.csv", index=False, encoding="utf-8-sig")

    # 5-class distribution
    table5 = make_distribution(df, GESTURE_ORDER_5)
    table5.to_csv(out_dir / "sample_distribution_5class.csv", encoding="utf-8-sig")

    # 4-class distribution
    table4 = make_distribution(df, GESTURE_ORDER_4)
    table4.to_csv(out_dir / "sample_distribution_4class.csv", encoding="utf-8-sig")

    # person count summary
    person_summary = (
        df.groupby(["person", "gesture", "head"])
        .size()
        .reset_index(name="count")
    )
    person_summary.to_csv(out_dir / "sample_distribution_by_person_gesture_head.csv",
                          index=False, encoding="utf-8-sig")

    latex5 = latex_table(
        table5,
        caption=(
            "Sample distribution across gesture classes and head-motion conditions "
            "in the five-class dataset. Each value indicates the number of segmented CSV trials."
        ),
        label="tab:supp_sample_distribution_5class"
    )

    latex4 = latex_table(
        table4,
        caption=(
            "Sample distribution across gesture classes and head-motion conditions "
            "in the four-class dataset. Each value indicates the number of segmented CSV trials."
        ),
        label="tab:supp_sample_distribution_4class"
    )

    with open(out_dir / "sample_distribution_5class_latex.tex", "w", encoding="utf-8") as f:
        f.write(latex5)

    with open(out_dir / "sample_distribution_4class_latex.tex", "w", encoding="utf-8") as f:
        f.write(latex4)

    with open(out_dir / "sample_distribution_tables_for_supplementary.tex", "w", encoding="utf-8") as f:
        f.write("% Four-class sample distribution\n")
        f.write(latex4)
        f.write("\n\n% Five-class sample distribution\n")
        f.write(latex5)

    print("=" * 80)
    print("[DONE] Sample distribution tables generated")
    print("=" * 80)
    print(f"Data root : {data_root}")
    print(f"Output dir: {out_dir}")
    print(f"Total CSV : {len(df)}")
    print()
    print("[5-class distribution]")
    print(table5)
    print()
    print("[4-class distribution]")
    print(table4)
    print()
    print("Generated files:")
    print(f"  {out_dir / 'sample_distribution_5class.csv'}")
    print(f"  {out_dir / 'sample_distribution_4class.csv'}")
    print(f"  {out_dir / 'sample_distribution_tables_for_supplementary.tex'}")

if __name__ == "__main__":
    main()
PY