#!/usr/bin/env python3
"""
MT Combined Evaluation Metric
==============================
Combines a standard MT metric (e.g. Blaser, scale 1-5) with perplexity
using log-normalized perplexity as a confidence weight.

Formula:
    w      = exp(-(log(PPL) - mu_log) / sigma_log)   # fluency confidence
    Score  = Blaser * w^alpha                          # penalized score
    Final  = rescaled to [1, 5]

Usage:
    python mt_combined_metric.py scores.tsv
    python mt_combined_metric.py scores.tsv --alpha 0.3 --no-rescale
    python mt_combined_metric.py scores.tsv --alpha 0.5 --output results.tsv
"""

import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────
#  Core metric functions
# ─────────────────────────────────────────────

def normalize_perplexity(ppl: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Convert raw perplexity values to a [0, 1] confidence weight.

    Parameters
    ----------
    ppl    : array of perplexity values (must be > 0)
    method : 'zscore'  — z-score on log(PPL), then exp-transform  [default]
             'minmax'  — min-max on log(PPL), then exp-transform
             'robust'  — percentile-based (1st–99th) min-max, robust to outliers

    Returns
    -------
    w : array of weights in (0, 1]; higher = more fluent
    """
    log_ppl = np.log(ppl)

    if method == "zscore":
        mu    = log_ppl.mean()
        sigma = log_ppl.std(ddof=1)
        if sigma == 0:
            return np.ones_like(log_ppl)
        w = np.exp(-(log_ppl - mu) / sigma)

    elif method == "minmax":
        lo, hi = log_ppl.min(), log_ppl.max()
        if hi == lo:
            return np.ones_like(log_ppl)
        w = np.exp(-((log_ppl - lo) / (hi - lo)))

    elif method == "robust":
        lo  = np.percentile(log_ppl, 1)
        hi  = np.percentile(log_ppl, 99)
        clipped = np.clip(log_ppl, lo, hi)
        if hi == lo:
            return np.ones_like(log_ppl)
        w = np.exp(-((clipped - lo) / (hi - lo)))

    else:
        raise ValueError(f"Unknown method '{method}'. Choose zscore | minmax | robust.")

    # Clip to (0, 1] — zscore can produce values > 1 for very low PPL
    w = np.clip(w, 1e-9, 1.0)
    return w


def combine_scores(
    blaser:   np.ndarray,
    ppl:      np.ndarray,
    alpha:    float = 0.5,
    method:   str   = "zscore",
    rescale:  bool  = True,
    out_min:  float = 1.0,
    out_max:  float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine Blaser and perplexity into a single metric.

    Score_raw   = Blaser * w^alpha
    Score_final = rescaled to [out_min, out_max]  (if rescale=True)

    Returns
    -------
    (raw_scores, final_scores)
    """
    w         = normalize_perplexity(ppl, method=method)
    raw       = blaser * (w ** alpha)

    if rescale:
        s_min  = raw.min()
        s_max  = raw.max()
        if s_max == s_min:
            final = np.full_like(raw, (out_min + out_max) / 2)
        else:
            final = out_min + (out_max - out_min) * (raw - s_min) / (s_max - s_min)
    else:
        final = raw.copy()

    return raw, final


# ─────────────────────────────────────────────
#  Statistics & validation helpers
# ─────────────────────────────────────────────

def descriptive_stats(arr: np.ndarray, name: str) -> dict:
    return {
        "metric":  name,
        "n":       len(arr),
        "mean":    arr.mean(),
        "std":     arr.std(ddof=1),
        "min":     arr.min(),
        "p25":     np.percentile(arr, 25),
        "median":  np.median(arr),
        "p75":     np.percentile(arr, 75),
        "max":     arr.max(),
        "skew":    float(stats.skew(arr)),
        "kurt":    float(stats.kurtosis(arr)),
    }


def correlation_report(blaser, ppl, combined):
    """Pearson and Spearman correlations between all metric pairs."""
    pairs = [
        ("Blaser",    "Perplexity", blaser,   ppl),
        ("Blaser",    "Combined",   blaser,   combined),
        ("Perplexity","Combined",   ppl,      combined),
    ]
    rows = []
    for a, b, x, y in pairs:
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        rows.append({
            "pair":             f"{a} ↔ {b}",
            "pearson_r":        round(pr, 4),
            "pearson_p":        round(pp, 4),
            "spearman_rho":     round(sr, 4),
            "spearman_p":       round(sp, 4),
        })
    return pd.DataFrame(rows)


def alpha_sensitivity(blaser, ppl, method="zscore", steps=10):
    """Show how the combined score distribution shifts across alpha values."""
    alphas = np.linspace(0, 1, steps + 1)
    rows = []
    for a in alphas:
        _, final = combine_scores(blaser, ppl, alpha=a, method=method, rescale=True)
        rows.append({
            "alpha":  round(a, 2),
            "mean":   round(final.mean(), 4),
            "std":    round(final.std(ddof=1), 4),
            "min":    round(final.min(), 4),
            "max":    round(final.max(), 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  I/O
# ─────────────────────────────────────────────

def load_tsv(path: str) -> pd.DataFrame:
    """
    Load a 2-column TSV.
    Accepts headers or no headers; first numeric column = Blaser, second = PPL.
    """
    try:
        df = pd.read_csv(path, sep="\t", header=None)
    except Exception as e:
        sys.exit(f"[ERROR] Could not read file '{path}': {e}")

    # Drop header row if present (non-numeric first row)
    if not pd.to_numeric(df.iloc[0, 0], errors="coerce") == df.iloc[0, 0]:
        df = df.iloc[1:].reset_index(drop=True)

    # Try reading with actual header
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] < 2:
            raise ValueError
        # Rename to standard names
        df.columns = ["blaser", "perplexity"] + list(df.columns[2:])
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None, names=["blaser", "perplexity"])

    df["blaser"]     = pd.to_numeric(df["blaser"],     errors="coerce")
    df["perplexity"] = pd.to_numeric(df["perplexity"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["blaser", "perplexity"])
    n_after  = len(df)

    if n_before != n_after:
        print(f"[WARN] Dropped {n_before - n_after} rows with missing values.")

    # Validate ranges
    if (df["blaser"] < 0).any():
        print("[WARN] Some Blaser scores are negative — check your data.")
    if (df["perplexity"] <= 0).any():
        sys.exit("[ERROR] Perplexity must be > 0. Found zero or negative values.")

    return df.reset_index(drop=True)


def print_section(title: str):
    w = 60
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combine an MT metric (Blaser) with perplexity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",
        help="TSV file with two columns: blaser_score, perplexity")
    parser.add_argument("--alpha", type=float, default=0.5,
        help="Perplexity influence [0=ignore PPL, 1=full weight] (default: 0.5)")
    parser.add_argument("--method", choices=["zscore", "minmax", "robust"],
        default="zscore",
        help="PPL normalization method (default: zscore)")
    parser.add_argument("--no-rescale", action="store_true",
        help="Do not rescale combined score back to [1, 5]")
    parser.add_argument("--output", type=str, default=None,
        help="Path to save results TSV (optional)")
    parser.add_argument("--sensitivity", action="store_true",
        help="Print alpha sensitivity analysis")

    args = parser.parse_args()

    # ── Load ──────────────────────────────────
    df = load_tsv(args.input)
    blaser = df["blaser"].to_numpy(dtype=float)
    ppl    = df["perplexity"].to_numpy(dtype=float)

    print_section(f"Loaded {len(df)} samples from '{args.input}'")
    print(f"  Blaser     range : [{blaser.min():.3f}, {blaser.max():.3f}]")
    print(f"  Perplexity range : [{ppl.min():.3f}, {ppl.max():.3f}]")

    # ── Normalize & combine ───────────────────
    w         = normalize_perplexity(ppl, method=args.method)
    raw, final = combine_scores(
        blaser, ppl,
        alpha   = args.alpha,
        method  = args.method,
        rescale = not args.no_rescale,
    )

    df["ppl_weight"]       = np.round(w, 6)
    df["combined_raw"]     = np.round(raw, 6)
    df["combined_final"]   = np.round(final, 6)

    # ── Descriptive stats ─────────────────────
    print_section("Descriptive Statistics")
    stat_rows = [
        descriptive_stats(blaser,           "Blaser (input)"),
        descriptive_stats(ppl,              "Perplexity (input)"),
        descriptive_stats(w,                "PPL weight [0-1]"),
        descriptive_stats(raw,              "Combined (raw)"),
        descriptive_stats(final,            "Combined (final)"),
    ]
    stats_df = pd.DataFrame(stat_rows).set_index("metric")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(stats_df.to_string())

    # ── Correlation report ────────────────────
    print_section("Correlation Report")
    corr_df = correlation_report(blaser, ppl, final)
    print(corr_df.to_string(index=False))

    # ── Settings summary ──────────────────────
    print_section("Configuration Used")
    print(f"  alpha            : {args.alpha}")
    print(f"  PPL norm method  : {args.method}")
    print(f"  Rescale to [1,5] : {not args.no_rescale}")
    print(f"  Formula          : Score = Blaser × w^{args.alpha}  "
          f"where w = exp(-(logPPL - μ) / σ)")

    # ── Alpha sensitivity ─────────────────────
    if args.sensitivity:
        print_section("Alpha Sensitivity Analysis")
        sens_df = alpha_sensitivity(blaser, ppl, method=args.method)
        print(sens_df.to_string(index=False))

    # ── Sample preview ────────────────────────
    print_section("Sample Output (first 10 rows)")
    preview = df[["blaser", "perplexity", "ppl_weight",
                  "combined_raw", "combined_final"]].head(10)
    print(preview.to_string(index=True))

    # ── Final average score ───────────────────
    n           = len(final)
    mean_score  = final.mean()
    sem_score   = final.std(ddof=1) / np.sqrt(n)          # standard error of the mean
    ci_lo, ci_hi = stats.t.interval(                       # 95 % confidence interval
        0.95, df=n - 1, loc=mean_score, scale=sem_score
    )
    # Arithmetic mean of Blaser alone (baseline for comparison)
    mean_blaser = blaser.mean()
    delta       = mean_score - mean_blaser
    pct_change  = (delta / mean_blaser) * 100

    print_section("★  Final Average Score")
    print(f"  Combined score (mean)    : {mean_score:.4f}  [scale 1–5]")
    print(f"  95 % CI                  : [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Std dev                  : {final.std(ddof=1):.4f}")
    print(f"  Std error of the mean    : {sem_score:.4f}")
    print(f"  N segments               : {n}")
    print()
    print(f"  Baseline Blaser (mean)   : {mean_blaser:.4f}")
    print(f"  Δ vs Blaser              : {delta:+.4f}  ({pct_change:+.2f} %)")
    print()
    # Qualitative band
    bands = [(1.0, 2.0, "Poor"), (2.0, 3.0, "Fair"),
             (3.0, 4.0, "Good"), (4.0, 5.0, "Very good"), (5.0, 5.01, "Excellent")]
    label = next((l for lo, hi, l in bands if lo <= mean_score < hi), "—")
    print(f"  Quality band             : {label}")
    print(f"{'═' * 60}")

    # ── Save ──────────────────────────────────
    if args.output:
        df.to_csv(args.output, sep="\t", index=False)
        print(f"\n✓ Full results saved to: {args.output}")
    else:
        print("\n(Use --output results.tsv to save the full table)")


if __name__ == "__main__":
    main()
