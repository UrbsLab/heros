#!/usr/bin/env python3
# job_heros_alt_table2.0.py
#
# Builds a combined HEROS table from dataset-level summary outputs:
#   <output_root>/HEROS_<group>/<dataset>/{mean_seed_evaluation_summary.csv, sd_seed_evaluation_summary.csv}
#
# Adds significance markers (*+/*-) by comparing each non-baseline config against a
# per-dataset-family baseline (multiplexer vs gametes), using raw eval CSVs:
#   all_<condition>_evaluations.csv (preferred) or cv_ave_<condition>_evaluations.csv (fallback)
#
# Usage (direct):
#   python job_heros_alt_table2.0.py --o /path/to/output --groups mux_cv_default,... \
#       --baseline-map multiplexer:mux_cv_default,gametes:gametes_cv_default --alpha 0.05 \
#       --outcsv HEROS_Combined_Table.csv

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests



# ----------------------------
# Table content configuration
# ----------------------------

TARGET_ROWS = [
    "default_model_200",
    "test_selected_model_200",
]

METRICS = {
    "Testing Accuracy": ("test_balanced_accuracy", 3),
    "Testing Coverage": ("test_coverage", 3),
    "Rule Count": ("rule_count", 1),
    "Run Time (Minutes)": ("run_time", 1),  # seconds -> minutes
}

PHASE1_CONDITIONS = [
    "rule_500", "rule_1000", "rule_10000", "rule_100000", "rule_200000", "rule_post_compact"
]
PHASE2_DEFAULT_CONDITIONS = [
    "default_model_5", "default_model_10", "default_model_50", "default_model_100", "default_model_200"
]
PHASE2_TESTSEL_CONDITIONS = [
    "test_selected_model_5", "test_selected_model_10", "test_selected_model_50",
    "test_selected_model_100", "test_selected_model_200"
]

GAMETES_DATASETS = [
    "A_uni_4add",
    "B_univariate",
    "C_2way_epistasis",
    "D_2way_epi_2het",
    "E_uni_4het",
    "F_3way_epistasis",
]

MUX_DATASETS = [
    "A_multiplexer_6_bit_500_inst",
    "B_multiplexer_11_bit_5000_inst",
    "C_multiplexer_20_bit_10000_inst",
    "D_multiplexer_37_bit_10000_inst",
    "E_multiplexer_70_bit_20000_inst",
]

IDEAL_MUX_RULE_COUNT = {
    "A_multiplexer_6_bit_500_inst": 8,
    "B_multiplexer_11_bit_5000_inst": 16,
    "C_multiplexer_20_bit_10000_inst": 32,
    "D_multiplexer_37_bit_10000_inst": 64,
    "E_multiplexer_70_bit_20000_inst": 128,
}


BASELINE_CONDITION = "default_model_200"

# ----------------------------
# Formatting / I/O helpers
# ----------------------------

def fmt_mean_sd(mean, sd, decimals=3):
    if mean is None or sd is None:
        return "NaN"
    try:
        if np.isnan(mean) or np.isnan(sd):
            return "NaN"
    except Exception:
        pass
    return f"{float(mean):.{decimals}f}({float(sd):.{decimals}f})"


def read_mean_sd(mean_path: Path, sd_path: Path) -> pd.DataFrame:
    """
    Reads:
      mean_seed_evaluation_summary.csv
      sd_seed_evaluation_summary.csv

    Expected format:
      Row Indexes, test_balanced_accuracy, ..., run_time
      rule_500, ...
      ...
    """
    if (not mean_path.exists()) or (not sd_path.exists()):
        return pd.DataFrame()

    mean_df = pd.read_csv(mean_path)
    sd_df = pd.read_csv(sd_path)

    if "Row Indexes" in mean_df.columns:
        mean_df = mean_df.rename(columns={"Row Indexes": "Condition"})
    if "Row Indexes" in sd_df.columns:
        sd_df = sd_df.rename(columns={"Row Indexes": "Condition"})

    if "Condition" not in mean_df.columns or "Condition" not in sd_df.columns:
        return pd.DataFrame()

    merged = pd.merge(mean_df, sd_df, on="Condition", how="outer", suffixes=("_mean", "_sd"))
    return merged


def raw_eval_path(group_dir: Path, dataset: str, condition: str) -> Path | None:
    """
    Prefer all_*_evaluations.csv; fallback to cv_ave_*_evaluations.csv.
    """
    p_all = group_dir / dataset / f"all_{condition}_evaluations.csv"
    if p_all.exists():
        return p_all

    p_cv = group_dir / dataset / f"cv_ave_{condition}_evaluations.csv"
    if p_cv.exists():
        return p_cv

    return None


def _metric_series_from_raw(df: pd.DataFrame, metric_col: str) -> pd.Series:
    if metric_col not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    return s.astype(float)


def load_raw_metric_values(
    output_root: Path,
    group_name: str,  # WITHOUT "HEROS_" prefix
    dataset: str,
    condition: str,
    metric_col: str,
) -> pd.Series:
    group_dir = output_root / f"HEROS_{group_name}"
    if not group_dir.exists():
        return pd.Series(dtype=float)

    p = raw_eval_path(group_dir, dataset, condition)
    if p is None:
        return pd.Series(dtype=float)

    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.Series(dtype=float)

    s = _metric_series_from_raw(df, metric_col)

    # match table display: runtime seconds -> minutes
    if metric_col == "run_time" and not s.empty:
        s = s / 60.0

    return s


def wilcoxon_p_or_fallback(base: pd.Series, other: pd.Series) -> tuple[float | None, str]:
    """
    Returns (p_value, test_name)
    - Wilcoxon (paired) if lengths match
    - else Mann-Whitney U (unpaired) fallback
    """
    base = pd.to_numeric(base, errors="coerce").dropna()
    other = pd.to_numeric(other, errors="coerce").dropna()

    if base.empty or other.empty:
        return (None, "none")

    if len(base) == len(other):
        try:
            _stat, p = wilcoxon(base.to_numpy(), other.to_numpy())
            return (float(p), "wilcoxon")
        except Exception:
            pass

    try:
        _stat, p = mannwhitneyu(base.to_numpy(), other.to_numpy(), alternative="two-sided")
        return (float(p), "mannwhitneyu")
    except Exception:
        return (None, "none")
    
def collect_significance_tests(
    output_root: Path,
    allowed_groups: set[str],
    *,
    baseline_map: dict[str, str],
) -> pd.DataFrame:
    rows = []

    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    group_dirs = []
    for g in sorted(allowed_groups):
        gp = output_root / f"HEROS_{g}"
        if gp.exists() and gp.is_dir():
            group_dirs.append(gp)

    for group_dir in group_dirs:
        group_name = group_dir.name.replace("HEROS_", "", 1)

        for dataset_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            dataset = dataset_dir.name
            fam = dataset_family(dataset)
            baseline_group = baseline_map.get(fam, None)

            if baseline_group is None:
                continue

            for condition in TARGET_ROWS:
                # skip the actual baseline row itself
                if group_name == baseline_group and condition == BASELINE_CONDITION:
                    continue

                for nice_name, (base_col, _decimals) in METRICS.items():
                    base_vec = load_raw_metric_values(
                        output_root, baseline_group, dataset, BASELINE_CONDITION, base_col
                    )
                    this_vec = load_raw_metric_values(
                        output_root, group_name, dataset, condition, base_col
                    )

                    if base_vec.empty or this_vec.empty:
                        continue

                    p_val, test_name = wilcoxon_p_or_fallback(base_vec, this_vec)
                    if p_val is None:
                        continue

                    base_mean = float(base_vec.mean()) if not base_vec.empty else None
                    this_mean = float(this_vec.mean()) if not this_vec.empty else None

                    rows.append({
                        "Dataset": dataset,
                        "DatasetFamily": fam,
                        "GroupName": group_name,
                        "Condition": condition,
                        "MetricLabel": nice_name,   # e.g. "Testing Accuracy"
                        "MetricCol": base_col,      # e.g. "test_balanced_accuracy"
                        "BaselineGroup": baseline_group,
                        "p_value": p_val,
                        "test_name": test_name,
                        "base_mean": base_mean,
                        "this_mean": this_mean,
                    })

    return pd.DataFrame(rows)

def apply_fdr_bh_by_metric(tests_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    tests_df = tests_df.copy()
    tests_df["p_value_fdr_bh"] = np.nan
    tests_df["significant"] = False

    if tests_df.empty:
        return tests_df

    for metric in tests_df["MetricLabel"].unique():
        metric_mask = tests_df["MetricLabel"] == metric
        metric_pvals = tests_df.loc[metric_mask, "p_value"].to_numpy()

        rej, p_adj, _, _ = multipletests(metric_pvals, alpha=alpha, method="fdr_bh")
        tests_df.loc[metric_mask, "p_value_fdr_bh"] = p_adj
        tests_df.loc[metric_mask, "significant"] = rej

    return tests_df

def build_sig_lookup(tests_df: pd.DataFrame) -> dict[tuple[str, str, str, str], dict]:
    lookup = {}

    for _, row in tests_df.iterrows():
        key = (
            row["Dataset"],
            row["GroupName"],
            row["Condition"],
            row["MetricLabel"],
        )
        lookup[key] = {
            "p_value": row["p_value"],
            "p_value_fdr_bh": row["p_value_fdr_bh"],
            "significant": bool(row["significant"]),
            "base_mean": row["base_mean"],
            "this_mean": row["this_mean"],
            "test_name": row["test_name"],
        }

    return lookup

def append_sig_marker(
    formatted_mean_sd: str,
    base_mean: float | None,
    this_mean: float | None,
    is_sig: bool,
    p_value_adj: float | None,
) -> str:
    if p_value_adj is None:
        return formatted_mean_sd

    p_str = f"{p_value_adj:.4g}"

    if not is_sig:
        return f"{formatted_mean_sd} ({p_str})"

    if base_mean is None or this_mean is None:
        return f"{formatted_mean_sd}* ({p_str})"

    return f"{formatted_mean_sd}{'*+' if this_mean > base_mean else '*-'} ({p_str})"


def parse_baseline_map(s: str) -> dict[str, str]:
    """
    "multiplexer:mux_cv_default,gametes:gametes_cv_default" -> {"multiplexer": "...", "gametes": "..."}
    """
    out: dict[str, str] = {}
    for chunk in (s or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            continue
        k, v = chunk.split(":", 1)
        out[k.strip().lower()] = v.strip()
    return out


def dataset_family(dataset: str) -> str:
    if dataset in MUX_DATASETS or "multiplexer" in dataset:
        return "multiplexer"
    if dataset in GAMETES_DATASETS:
        return "gametes"
    return "other"


def extract_one_condition(
    merged: pd.DataFrame,
    condition: str,
    *,
    output_root: Path,
    dataset: str,
    group_name: str,                 # WITHOUT "HEROS_"
    baseline_map: dict[str, str],
    alpha: float,
    sig_lookup: dict | None = None,
) -> dict:
    out = {"Condition": condition}
    out["Ideal Solution Count"] = ""

    if merged.empty:
        for nice_name in METRICS:
            out[nice_name] = "NaN"
        return out

    row = merged[merged["Condition"] == condition]
    if row.empty:
        for nice_name in METRICS:
            out[nice_name] = "NaN"
        return out

    row0 = row.iloc[0]
    # Only for MUX configs + Phase II model rows
    out["Ideal Solution Count"] = ideal_solution_count_cell(
        output_root=output_root,
        group_name=group_name,
        dataset=dataset,
        condition=condition,
    )


    fam = dataset_family(dataset)
    baseline_group = baseline_map.get(fam, None)
    do_sig = (
        baseline_group is not None and
        not (group_name == baseline_group and condition == BASELINE_CONDITION)
    )

    for nice_name, (base_col, decimals) in METRICS.items():
        mean_col = f"{base_col}_mean"
        sd_col = f"{base_col}_sd"
        mean_val = row0.get(mean_col, None)
        sd_val = row0.get(sd_col, None)

        # runtime seconds -> minutes (for display)
        if base_col == "run_time":
            mean_val = (float(mean_val) / 60.0) if mean_val is not None else None
            sd_val = (float(sd_val) / 60.0) if sd_val is not None else None

        cell = fmt_mean_sd(mean_val, sd_val, decimals=decimals)

        if do_sig and sig_lookup is not None:
            key = (dataset, group_name, condition, nice_name)
            sig_info = sig_lookup.get(key)

            if sig_info is not None:
                cell = append_sig_marker(
                    cell,
                    sig_info.get("base_mean"),
                    sig_info.get("this_mean"),
                    bool(sig_info.get("significant")),
                    sig_info.get("p_value_fdr_bh"),
                )

        out[nice_name] = cell

    return out

IDEAL_DENOM = 200  # fixed per your design (not inferred from file length)

def is_mux_phase2_model_condition(condition: str) -> bool:
    return condition in PHASE2_DEFAULT_CONDITIONS or condition in PHASE2_TESTSEL_CONDITIONS


def ideal_solution_count_cell(
    output_root: Path,
    group_name: str,   # WITHOUT "HEROS_"
    dataset: str,
    condition: str,
) -> str:
    """
    Returns 'X/200' for MUX Phase II model rows, else ''.
    Uses raw eval CSV (all_* preferred; cv_ave_* fallback).
    """
    if dataset not in IDEAL_MUX_RULE_COUNT:
        return ""
    if not is_mux_phase2_model_condition(condition):
        return ""

    group_dir = output_root / f"HEROS_{group_name}"
    p = raw_eval_path(group_dir, dataset, condition)
    if p is None or not p.exists():
        return ""

    try:
        df = pd.read_csv(p)
    except Exception:
        return ""

    # Required columns in raw eval files
    if "test_balanced_accuracy" not in df.columns or "rule_count" not in df.columns:
        return ""

    acc = pd.to_numeric(df["test_balanced_accuracy"], errors="coerce")
    rc = pd.to_numeric(df["rule_count"], errors="coerce")

    ideal_rc = IDEAL_MUX_RULE_COUNT[dataset]

    # acc == 1.0 with tolerance; rule_count matches ideal after rounding
    hits = ((acc >= 1.0 - 1e-12) & (rc.round() == ideal_rc)).sum()

    return f"{int(hits)}/{IDEAL_DENOM}"



# ----------------------------
# Table builder
# ----------------------------

def build_table(output_root: Path, allowed_groups: set[str], *, baseline_map: dict[str, str], alpha: float, sig_lookup: dict | None = None) -> pd.DataFrame:
    rows = []

    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    # Only iterate over the selected groups (and only if they exist)
    group_dirs = []
    for g in sorted(allowed_groups):
        gp = output_root / f"HEROS_{g}"
        if gp.exists() and gp.is_dir():
            group_dirs.append(gp)

    if not group_dirs:
        return pd.DataFrame()

    for group_dir in group_dirs:
        group_name = group_dir.name.replace("HEROS_", "", 1)

        for dataset_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            mean_path = dataset_dir / "mean_seed_evaluation_summary.csv"
            sd_path = dataset_dir / "sd_seed_evaluation_summary.csv"
            merged = read_mean_sd(mean_path, sd_path)
            if merged.empty:
                continue

            for cond in TARGET_ROWS:
                record = {
                    "Dataset": dataset_dir.name,
                    "Algorithm Config.": group_dir.name,
                }
                record.update(extract_one_condition(
                    merged,
                    cond,
                    output_root=output_root,
                    dataset=dataset_dir.name,
                    group_name=group_name,
                    baseline_map=baseline_map,
                    alpha=alpha,
                    sig_lookup=sig_lookup,
                ))
                rows.append(record)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    col_order = ["Dataset", "Algorithm Config.", "Condition", "Ideal Solution Count"] + list(METRICS.keys())
    df = df[col_order]

    cond_order = {c: i for i, c in enumerate(TARGET_ROWS)}
    df["__cond_order"] = df["Condition"].map(cond_order).fillna(10_000).astype(int)

    df = df.sort_values(
        by=["Dataset", "Algorithm Config.", "__cond_order"],
        ascending=[True, True, True],
        kind="mergesort"
    ).drop(columns="__cond_order").reset_index(drop=True)

    return df


# ----------------------------
# Plot helpers (unchanged)
# ----------------------------

def collect_boxplot_data(output_root: Path, allowed_groups: set[str], conditions: list[str], metric_col: str) -> pd.DataFrame:
    rows = []
    for g in sorted(allowed_groups):
        group_dir = output_root / f"HEROS_{g}"
        if not group_dir.exists():
            continue

        for dataset_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            dataset = dataset_dir.name
            for cond in conditions:
                pth = raw_eval_path(group_dir, dataset, cond)
                if pth is None:
                    continue
                try:
                    df = pd.read_csv(pth)
                except Exception:
                    continue
                if metric_col not in df.columns:
                    continue
                for v in pd.to_numeric(df[metric_col], errors="coerce").dropna().tolist():
                    rows.append({
                        "Dataset": dataset,
                        "Algorithm Config.": group_dir.name,
                        "Condition": cond,
                        "Value": float(v),
                    })
    return pd.DataFrame(rows)


def format_dataset_label(ds: str) -> str:
    if "multiplexer" in ds:
        try:
            return ds.split("_multiplexer_")[1].split("_bit")[0]
        except Exception:
            return ds
    return ds.split("_")[0]


def pretty_config_label(cfg: str) -> str:
    if cfg.endswith("default"):
        return "HEROS"
    if cfg.endswith("default_tree_bstrap"):
        return "HEROS-Tree"
    if cfg.endswith("equal_tree_bstrap"):
        return "HEROS-Tree-Alt"
    return cfg


def make_grouped_boxplot_by_dataset(
    long_df: pd.DataFrame,
    out_png: Path,
    ylabel: str,
    conditions_in_order: list[str],
    figsize=(15, 7),
):
    if long_df.empty:
        print(f"[WARN] No data for plot: {out_png.name}")
        return

    DISTINCT_COLORS = [
        "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    datasets = sorted(long_df["Dataset"].unique())
    configs = sorted(long_df["Algorithm Config."].unique())
    present_conditions = set(long_df["Condition"].unique())
    conditions = [c for c in conditions_in_order if c in present_conditions]

    box_width = 0.05
    group_spacing = 0.05
    dataset_gap = 0.10

    hatch_cycle = ["", "///", "..."]
    hatch_by_config = {cfg: hatch_cycle[i % len(hatch_cycle)] for i, cfg in enumerate(configs)}

    if len(conditions) > len(DISTINCT_COLORS):
        raise ValueError(f"Not enough distinct colors for {len(conditions)} conditions")
    color_by_condition = {cond: DISTINCT_COLORS[i] for i, cond in enumerate(conditions)}

    fig, ax = plt.subplots(figsize=figsize)

    base_x = 0.0
    xticks, xtick_labels = [], []
    x_positions_used = []
    last_block_top_anchor_x = None

    for ds_i, ds in enumerate(datasets):
        n_groups = len(conditions) * len(configs)
        block_width = (n_groups - 1) * group_spacing
        block_right = base_x + block_width
        block_center = base_x + block_width / 2.0

        xticks.append(block_center)
        xtick_labels.append(format_dataset_label(ds))

        g_i = 0
        for cond in conditions:
            for cfg in configs:
                sub = long_df[
                    (long_df["Dataset"] == ds) &
                    (long_df["Condition"] == cond) &
                    (long_df["Algorithm Config."] == cfg)
                ]["Value"].to_numpy()

                if sub.size > 0:
                    x = base_x + g_i * group_spacing
                    x_positions_used.append(x)
                    bp = ax.boxplot(
                        [sub],
                        positions=[x],
                        widths=box_width,
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker="o", markerfacecolor="black", markeredgecolor="white", markersize=4),
                        medianprops=dict(color="yellow", linewidth=1.5),
                    )

                    for box in bp["boxes"]:
                        box.set_facecolor(color_by_condition[cond])
                        box.set_hatch(hatch_by_config[cfg])
                        box.set_edgecolor("black")
                        box.set_linewidth(1.0)

                    for w in bp["whiskers"]:
                        w.set_color("black")
                        w.set_linewidth(1.0)
                    for c in bp["caps"]:
                        c.set_color("black")
                        c.set_linewidth(1.0)

                g_i += 1

        if ds_i == len(datasets) - 1:
            last_block_top_anchor_x = block_right - 0.5 * group_spacing

        base_x += block_width + dataset_gap

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(xticks)
    list = enumerate(datasets)
    check = next(list)[1]
    if "multiplexer" in check:
        ax.set_xlabel("MUX Dataset", fontsize=14)
    else: 
        ax.set_xlabel("GAMETES Dataset", fontsize=14)
    ax.set_xticklabels(xtick_labels, rotation=0, ha="center", fontsize=14)
    ax.tick_params(axis="y", labelsize=14)

    if x_positions_used:
        xmin = min(x_positions_used) - box_width / 2.0
        xmax = max(x_positions_used) + box_width / 2.0
        ax.set_xlim(xmin, xmax)
    ax.margins(x=0)

    legend_handles = []
    for cfg in configs:
        legend_handles.append(Patch(facecolor="white", edgecolor="black", hatch=hatch_by_config[cfg],
                                   label=pretty_config_label(cfg)))
    for cond in conditions:
        lab = cond.replace("default_model_", "").replace("test_selected_model_", "")
        lab = lab + " Iterations"
        legend_handles.append(Patch(facecolor=color_by_condition[cond], edgecolor="black", label=lab))

    if last_block_top_anchor_x is None:
        ax.legend(handles=legend_handles, loc="upper left", fontsize=14, frameon=True)
    else:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            #bbox_to_anchor=(last_block_top_anchor_x, 1.0),
            #bbox_transform=ax.transData,
            fontsize=13,
            frameon=True,
            borderaxespad=0.2,
            labelspacing=0.35,
            handlelength=1.8,
            handletextpad=0.6,
        )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"✅ Saved plot: {out_png}")


# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="Generate HEROS combined summary table from summary CSV outputs (with significance).")
    p.add_argument("--o", dest="outputPath", required=True,
                   help="Path to output directory (the folder containing HEROS_<group>/ folders).")
    p.add_argument("--groups", dest="groups", required=True,
                   help="Comma-separated group names to include (WITHOUT 'HEROS_' prefix).")
    p.add_argument("--outcsv", dest="outcsv", default="HEROS_Combined_Table.csv",
                   help="Output CSV file name (written inside --o).")
    p.add_argument("--baseline-map", dest="baseline_map",
                   default="multiplexer:mux_cv_default,gametes:gametes_cv_default",
                   help="Per-family baselines for significance. Format: 'multiplexer:<group>,gametes:<group>'.")
    p.add_argument("--alpha", dest="alpha", type=float, default=0.05,
                   help="Significance threshold p-value / FDR target (default 0.05).")

    args = p.parse_args()

    output_root = Path(args.outputPath).resolve()
    allowed_groups = {g.strip() for g in args.groups.split(",") if g.strip()}

    baseline_map = parse_baseline_map(args.baseline_map)
    alpha = args.alpha

    print(f"Using baseline map: {baseline_map} (alpha={alpha})")

    tests_df = collect_significance_tests(
        output_root,
        allowed_groups,
        baseline_map=baseline_map,
    )
    tests_df = apply_fdr_bh_by_metric(tests_df, alpha=alpha)
    sig_lookup = build_sig_lookup(tests_df)

    combined = build_table(
        output_root,
        allowed_groups,
        baseline_map=baseline_map,
        alpha=alpha,
        sig_lookup=sig_lookup,
    )

    if combined.empty:
        print(f"[WARN] No valid mean/sd summary CSVs found under: {output_root} for groups: {sorted(allowed_groups)}")
        return

    out_path = output_root / args.outcsv
    combined.to_csv(out_path, index=False)
    print(f"✅ Combined table saved to: {out_path}")

    tests_out_path = output_root / args.outcsv.replace(".csv", "_significance_tests_long.csv")
    tests_df.to_csv(tests_out_path, index=False)
    print(f"✅ Significance test details saved to: {tests_out_path}")

    # ---- ALWAYS generate boxplots ----
    base_plot_dir = output_root / "heros_boxplots"
    (base_plot_dir / "multiplexer").mkdir(parents=True, exist_ok=True)
    (base_plot_dir / "gametes").mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("Phase1", PHASE1_CONDITIONS, "Phase I Tracking"),
        ("Phase2_default", PHASE2_DEFAULT_CONDITIONS, "Phase II Default Model Tracking"),
        ("Phase2_testsel", PHASE2_TESTSEL_CONDITIONS, "Phase II Test-Selected Tracking"),
    ]

    splits = [
        ("multiplexer", MUX_DATASETS),
        ("gametes", GAMETES_DATASETS),
    ]

    for tag, conds, _pretty in plot_specs:
        for metric_name, (metric_col, _dec) in METRICS.items():
            df_long = collect_boxplot_data(output_root, allowed_groups, conds, metric_col)

            ylabel = metric_name
            if metric_col == "run_time" and not df_long.empty:
                df_long = df_long.copy()
                df_long["Value"] = df_long["Value"] / 60.0
                ylabel = "Run Time (Minutes)"

            for split_name, dataset_set in splits:
                df_split = df_long[df_long["Dataset"].isin(dataset_set)]
                out_png = base_plot_dir / split_name / f"{tag}_{metric_col}_boxplot.png"

                make_grouped_boxplot_by_dataset(
                    df_split,
                    out_png,
                    ylabel=ylabel,
                    conditions_in_order=conds,
                    figsize=(15, 7),
                )


if __name__ == "__main__":
    main()
