#!/usr/bin/env python3
"""
collect_evaluation_metrics.py

Aggregate RF-residual evaluation logs into a single wide CSV:
  - Each column: one county (e.g., G1300510)
  - Each row   : one metric
  - Raw and Enforced results share the same county column, distinguished by row names
    (e.g., "Raw Hourly MAE (C)" vs "Enforced Hourly MAE (C)").

Expected layout:
  MODEL_ROOT/Gxxxxxxx/raw/evaluation_log.txt
  MODEL_ROOT/Gxxxxxxx/enforced/evaluation_log.txt

Run:
  python collect_evaluation_metrics.py --model_root ... --output_csv ... --debug
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import pandas as pd


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

# Robust: allow "Test(2016-2024): 78912" or "Test: 87672"
RE_SPLIT = re.compile(
    r"Merged hours:\s*(?P<merged>\d+)\s*\|\s*Train:\s*(?P<train>\d+)\s*\|\s*Test.*?\s*:\s*(?P<test>\d+)",
    re.IGNORECASE,
)
RE_TRAIN_TIME = re.compile(r"Train time:\s*(?P<sec>" + FLOAT + r")s", re.IGNORECASE)

RE_MINMAX = re.compile(
    r"\[MinMaxEnforce\]\s*days=(?P<days>\d+),\s*scaled=(?P<scaled>\d+),\s*template_used=(?P<tpl>\d+),\s*skipped=(?P<skipped>\d+)",
    re.IGNORECASE,
)

RE_OVERALL = re.compile(
    r"Overall hourly.*\|\s*MAE=(?P<mae>" + FLOAT + r"),\s*RMSE=(?P<rmse>" + FLOAT + r"),\s*R2=(?P<r2>" + FLOAT + r")",
    re.IGNORECASE,
)
RE_KS = re.compile(
    r"KS test\s*\|\s*stat=(?P<stat>" + FLOAT + r"),\s*p-value=(?P<pval>" + FLOAT + r")",
    re.IGNORECASE,
)
RE_DAILY_VALUES = re.compile(
    r"Daily extreme VALUES\s*\|\s*Max MAE=(?P<max_mae>" + FLOAT + r"),\s*Max RMSE=(?P<max_rmse>" + FLOAT + r"),\s*Min MAE=(?P<min_mae>" + FLOAT + r"),\s*Min RMSE=(?P<min_rmse>" + FLOAT + r")",
    re.IGNORECASE,
)
# Robust: allow spaces around tokens, keep "Acc<=1h"
RE_DAILY_TIMING = re.compile(
    r"Daily extreme TIMING\s*\|\s*MaxHour MAE\(circ\)=(?P<maxh>" + FLOAT + r")\s*,\s*"
    r"MinHour MAE\(circ\)=(?P<minh>" + FLOAT + r")\s*,\s*"
    r"Acc<=1h\s*\(max/min\)=(?P<acc_max>" + FLOAT + r")\s*/\s*(?P<acc_min>" + FLOAT + r")",
    re.IGNORECASE,
)


def discover_logs(model_root: str) -> List[Tuple[str, str, str]]:
    """
    Return list of (county, variant, log_path) where variant in {"raw","enforced"}.
    """
    out: List[Tuple[str, str, str]] = []
    for variant in ["raw", "enforced"]:
        pattern = os.path.join(model_root, "G*", variant, "evaluation_log.txt")
        for p in sorted(glob.glob(pattern)):
            county = os.path.basename(os.path.dirname(os.path.dirname(p)))
            out.append((county, variant, p))
    return out


def metric_rows() -> List[str]:
    """
    Stable row order (concise names, no underscores).
    """
    base = [
        "Log Present",
        "Merged Hours",
        "Train Hours",
        "Test Hours (2016-2024)",
        "Train Time (s)",
        "Hourly MAE (C)",
        "Hourly RMSE (C)",
        "Hourly R2",
        "KS Stat",
        "KS P-value",
        "Daily Max MAE (C)",
        "Daily Max RMSE (C)",
        "Daily Min MAE (C)",
        "Daily Min RMSE (C)",
        "Max Hour MAE (h)",
        "Min Hour MAE (h)",
        "Acc<=1h (Max)",
        "Acc<=1h (Min)",
        "MinMax Days",
        "MinMax Scaled",
        "MinMax Template Used",
        "MinMax Skipped",
    ]
    rows: List[str] = []
    for prefix in ["Raw", "Enforced"]:
        rows.extend([f"{prefix} {b}" for b in base])
    return rows


def parse_log(log_path: str, label: str) -> Dict[str, float]:
    """
    Parse one evaluation_log.txt into metric dict with the given label.
    """
    metrics: Dict[str, float] = {}
    if not os.path.exists(log_path):
        return metrics

    # Read & strip, also remove Windows CR if present
    with open(log_path, "r") as f:
        lines = [ln.strip().replace("\r", "") for ln in f if ln.strip()]

    for ln in lines:
        m = RE_SPLIT.search(ln)
        if m:
            metrics[f"{label} Merged Hours"] = float(m.group("merged"))
            metrics[f"{label} Train Hours"] = float(m.group("train"))
            metrics[f"{label} Test Hours (2016-2024)"] = float(m.group("test"))
            continue

        m = RE_TRAIN_TIME.search(ln)
        if m:
            metrics[f"{label} Train Time (s)"] = float(m.group("sec"))
            continue

        m = RE_MINMAX.search(ln)
        if m:
            metrics[f"{label} MinMax Days"] = float(m.group("days"))
            metrics[f"{label} MinMax Scaled"] = float(m.group("scaled"))
            metrics[f"{label} MinMax Template Used"] = float(m.group("tpl"))
            metrics[f"{label} MinMax Skipped"] = float(m.group("skipped"))
            continue

        m = RE_OVERALL.search(ln)
        if m:
            metrics[f"{label} Hourly MAE (C)"] = float(m.group("mae"))
            metrics[f"{label} Hourly RMSE (C)"] = float(m.group("rmse"))
            metrics[f"{label} Hourly R2"] = float(m.group("r2"))
            continue

        m = RE_KS.search(ln)
        if m:
            metrics[f"{label} KS Stat"] = float(m.group("stat"))
            metrics[f"{label} KS P-value"] = float(m.group("pval"))
            continue

        m = RE_DAILY_VALUES.search(ln)
        if m:
            metrics[f"{label} Daily Max MAE (C)"] = float(m.group("max_mae"))
            metrics[f"{label} Daily Max RMSE (C)"] = float(m.group("max_rmse"))
            metrics[f"{label} Daily Min MAE (C)"] = float(m.group("min_mae"))
            metrics[f"{label} Daily Min RMSE (C)"] = float(m.group("min_rmse"))
            continue

        m = RE_DAILY_TIMING.search(ln)
        if m:
            metrics[f"{label} Max Hour MAE (h)"] = float(m.group("maxh"))
            metrics[f"{label} Min Hour MAE (h)"] = float(m.group("minh"))
            metrics[f"{label} Acc<=1h (Max)"] = float(m.group("acc_max"))
            metrics[f"{label} Acc<=1h (Min)"] = float(m.group("acc_min"))
            continue

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_root",
        default="/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/rf_residual_model",
        help="Root directory that contains Gxxxxxxx county folders.",
    )
    ap.add_argument(
        "--output_csv",
        default="/nfs/turbo/seas-mtcraig-climate/Ziqi/LOCA2/file/rf_residual_evaluation_summary.csv",
        help="Output CSV path.",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info (logs found, metrics matched).",
    )
    args = ap.parse_args()

    logs = discover_logs(args.model_root)
    if not logs:
        raise RuntimeError(
            f"No evaluation_log.txt found.\n"
            f"Expected: {args.model_root}/G*/raw/evaluation_log.txt and/or enforced/evaluation_log.txt"
        )

    counties = sorted({c for c, _, _ in logs})
    rows = metric_rows()

    df = pd.DataFrame(index=rows, columns=counties)

    # Presence flags (diagnostic)
    present = {(c, v): 0 for c in counties for v in ("raw", "enforced")}
    for c, v, _ in logs:
        present[(c, v)] = 1
    for c in counties:
        df.loc["Raw Log Present", c] = present[(c, "raw")]
        df.loc["Enforced Log Present", c] = present[(c, "enforced")]

    # Parse logs
    parsed_nonempty = 0
    for county, variant, path in logs:
        label = "Raw" if variant == "raw" else "Enforced"
        metrics = parse_log(path, label)

        if args.debug:
            print(f"[DEBUG] {county} {variant} -> matched {len(metrics)} metrics | {path}")

        if len(metrics) == 0 and args.debug:
            # Print a few key lines to diagnose regex mismatch quickly
            with open(path, "r") as f:
                sample = [ln.strip() for ln in f.readlines()[:30]]
            print("[DEBUG] First ~30 lines:")
            for s in sample:
                print("   ", s)

        for k, v in metrics.items():
            if k not in df.index:
                # Forward compatible: add row if log introduces new metric later
                df.loc[k, :] = pd.NA
            df.loc[k, county] = v

        if metrics:
            parsed_nonempty += 1

    # Keep stable order, append any new rows at end
    ordered = [r for r in rows if r in df.index]
    extra = [r for r in df.index if r not in rows]
    df = df.loc[ordered + extra]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df.to_csv(args.output_csv, index=True)

    print(f"Saved: {args.output_csv}")
    print(f"Counties: {len(counties)}")
    if args.debug:
        raw_logs = sum(present[(c, "raw")] for c in counties)
        enf_logs = sum(present[(c, "enforced")] for c in counties)
        print(f"[DEBUG] raw logs present: {raw_logs} | enforced logs present: {enf_logs}")
        print(f"[DEBUG] logs with >=1 matched metric: {parsed_nonempty} / {len(logs)}")
        print("If Log Present rows are filled but others are empty, regex did not match your log lines.")


if __name__ == "__main__":
    main()

