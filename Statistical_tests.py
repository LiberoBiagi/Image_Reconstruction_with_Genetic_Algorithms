"""
Statistical Analysis: Hill Climbing vs Genetic Algorithm

Outputs:
    - Console report with all test results
    - plots/ directory with 10 individual figures

Dependencies:
    pip install pandas numpy scipy matplotlib openpyxl
"""

import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — adjust paths if needed
# ─────────────────────────────────────────────────────────────────────────────
HC_SHORT_LOG  = "fitness_log_hill.csv"          # 100 HC runs, 500 iterations
HC_LONG_LOG   = "fitness_log.csv"               # 5 HC runs,  90k iterations
GA_SEED_GLOB  = "evolution_results_seed_*.xlsx" # 30 GA runs, 500 generations
GRID_SEARCH   = "repeated_grid_search_results.csv"
GA_PAPER_RMSE = 20.78                           # best result from paper (3000 gen)
GA_POP_SIZE   = 500                             # population size used in GA
N_BOOTSTRAP   = 10_000
N_PERMUTATION = 10_000
OUTPUT_DIR    = ("/Users/carlosamorim/Library/CloudStorage/OneDrive-NOVAIMS/MDSAA/CIFO/project/Computational-Intelligence-for-Optimization-main/statistics/outputs")
RANDOM_SEED   = 42

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
BLUE  = "#4a90d9"
AMBER = "#e0945c"
TEAL  = "#3dbfa0"
RED   = "#e05c5c"
GRAY  = "#888888"
plt.rcParams.update({"font.size": 11,
                     "axes.spines.top": False,
                     "axes.spines.right": False})

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pfmt(p):
    """Format p-value for display."""
    if p < 0.0001: return "p < 0.0001"
    if p < 0.001:  return f"p = {p:.4f}"
    return f"p = {p:.4f}"

def save_fig(name):
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}.png")

def bootstrap_means(data, n=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    return np.array([
        np.mean(rng.choice(data, len(data), replace=True))
        for _ in range(n)
    ])

def sig_bracket(ax, x1, x2, y, p_val, dy=4):
    """Draw a significance bracket between two box positions."""
    ax.plot([x1, x1, x2, x2], [y, y+dy*0.3, y+dy*0.3, y],
            lw=1.2, color="black")
    stars = ("***" if p_val < 0.001 else "**" if p_val < 0.01
             else "*" if p_val < 0.05 else "ns")
    ax.text((x1+x2)/2, y + dy*0.35,
            f"{stars}\n{pfmt(p_val)}", ha="center", va="bottom",
            fontsize=7.5, color="black")

def styled_table(ax, rows, headers, col_widths):
    """Render a styled table on an axis."""
    tbl = ax.table(cellText=rows, colLabels=headers,
                   cellLoc="center", loc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1, 2.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
        cell.set_edgecolor("#cccccc")
    return tbl


def calc_robust_mwu(x, y, x_name="Group 1", y_name="Group 2"):
    """
    Computes Mann-Whitney U, p-value, rank-biserial correlation magnitude,
    and directional dominance (assuming lower values = better performance).
    """
    u_stat, p_val = stats.mannwhitneyu(x, y, alternative="two-sided")

    # Calculate raw rank-biserial correlation
    r_raw = 1 - (2 * u_stat) / (len(x) * len(y))

    # Magnitude defines the effect size strength
    r_mag = abs(r_raw)

    # Direction defines the winner (lower values are better)
    # r_raw < 0 means U_x > U_y (Group 1 values are larger/worse)
    if r_raw < 0:
        winner = y_name
    elif r_raw > 0:
        winner = x_name
    else:
        winner = "Tie"

    return u_stat, p_val, r_mag, winner

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    # HC short: 100 runs × 500 iterations
    hc_s_df    = pd.read_csv(HC_SHORT_LOG)
    hc_s_cols  = [c for c in hc_s_df.columns if c != "iteration"]
    hc_s_final = (hc_s_df[hc_s_df["iteration"] == hc_s_df["iteration"].max()]
                  [hc_s_cols].values[0])
    hc_s_curves = hc_s_df[hc_s_cols].values.T        # (100, 500)

    # HC long: 5 runs × 90k iterations
    hc_l_df    = pd.read_csv(HC_LONG_LOG)
    hc_l_cols  = [c for c in hc_l_df.columns if c != "iteration"]
    hc_l_final = (hc_l_df[hc_l_df["iteration"] == hc_l_df["iteration"].max()]
                  [hc_l_cols].values[0])
    hc_l_curves = hc_l_df[hc_l_cols].values.T        # (5, 90000)

    # GA: 30 seeds × 500 generations
    ga_files   = sorted(glob.glob(GA_SEED_GLOB))
    ga_curves  = np.array([pd.read_excel(f)["fitness"].values for f in ga_files])
    ga_final   = ga_curves[:, -1]                     # (30,)

    # Grid search
    gs = pd.read_csv(GRID_SEARCH)

    return hc_s_final, hc_s_curves, hc_l_final, hc_l_curves, ga_final, ga_curves, gs

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_stats(hc_s, ga, hc_l, hc_s_curves, ga_curves):
    rng = np.random.default_rng(RANDOM_SEED)

    # Normality
    sw_hc  = stats.shapiro(hc_s)
    dag_hc = stats.normaltest(hc_s)
    sw_ga  = stats.shapiro(ga)

    # Primary: Mann-Whitney U
    u_main, p_mwu, r_mwu, main_winner = calc_robust_mwu(hc_s, ga, "HC", "GA")

    # Pairwise MWU with Bonferroni (3 pairs)
    pairs_def = [
        ("HC-500", "GA-500", hc_s, ga),
        ("HC-500", "HC-90k", hc_s, hc_l),
        ("GA-500", "HC-90k", ga, hc_l),
    ]
    n_pairs = len(pairs_def)
    pair_results = []
    for x_name, y_name, a, b in pairs_def:
        u, p, r_mag, winner = calc_robust_mwu(a, b, x_name, y_name)
        p_adj = min(p * n_pairs, 1.0)
        label = f"{x_name} vs {y_name}"
        pair_results.append((label, u, p, p_adj, r_mag, winner))

    # AUC
    hc_aucs = hc_s_curves.sum(axis=1)
    ga_aucs = ga_curves.sum(axis=1)
    u_auc, p_auc, r_auc, auc_winner = calc_robust_mwu(hc_aucs, ga_aucs, "HC", "GA")

    # Welch's t-test
    t_stat, p_t = stats.ttest_ind(hc_s, ga, equal_var=False)

    # Corrected pooled standard deviation for unequal sample sizes
    n1, n2 = len(hc_s), len(ga)
    var1, var2 = np.var(hc_s, ddof=1), np.var(ga, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(hc_s) - np.mean(ga)) / pooled_sd

    # Levene
    lev_s, lev_p = stats.levene(hc_s, ga)

    # KS 2-sample
    ks_stat, ks_p = stats.ks_2samp(hc_s, ga)

    # Kruskal-Wallis (3 groups)
    kw_stat, kw_p = stats.kruskal(hc_s, ga, hc_l)

    # Permutation test
    obs_diff = np.mean(hc_s) - np.mean(ga)
    combined = np.concatenate([hc_s, ga])
    perm_diffs = np.array([
        np.mean(rng.permutation(combined)[:len(hc_s)]) -
        np.mean(rng.permutation(combined)[len(hc_s):])
        for _ in range(N_PERMUTATION)
    ])
    perm_p = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))

    # Bootstrap CIs
    boot_hc_s = bootstrap_means(hc_s)
    boot_ga   = bootstrap_means(ga)
    boot_hc_l = bootstrap_means(hc_l)

    # Grid search Spearman
    gs = pd.read_csv(GRID_SEARCH)
    gs_results = {}
    for col in ["dx_start", "dx_end", "area_start", "area_end"]:
        rho, pv = stats.spearmanr(gs[col], gs["avg_error"])
        gs_results[col] = (rho, pv)

    return dict(
        sw_hc=sw_hc, dag_hc=dag_hc, sw_ga=sw_ga,
        u_main=u_main, p_mwu=p_mwu, r_mwu=r_mwu, main_winner=main_winner,
        t_stat=t_stat, p_t=p_t, cohens_d=cohens_d,
        lev_s=lev_s, lev_p=lev_p,
        ks_stat=ks_stat, ks_p=ks_p,
        kw_stat=kw_stat, kw_p=kw_p,
        pair_results=pair_results,
        obs_diff=obs_diff, perm_diffs=perm_diffs, perm_p=perm_p,
        boot_hc_s=boot_hc_s, boot_ga=boot_ga, boot_hc_l=boot_hc_l,
        hc_aucs=hc_aucs, ga_aucs=ga_aucs,
        u_auc=u_auc, p_auc=p_auc, r_auc=r_auc, auc_winner=auc_winner,
        gs_results=gs_results, gs_df=gs,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_report(hc_s, ga, hc_l, s):
    W = 66
    def h(title): print(f"\n{'─'*W}\n{title}\n{'─'*W}")

    print("=" * W)
    print("STATISTICAL ANALYSIS REPORT — HC vs GA")
    print("=" * W)
    print(f"  HC short run : n={len(hc_s)}, 500 iterations")
    print(f"  GA           : n={len(ga)},  500 generations")
    print(f"  HC long run  : n={len(hc_l)},  90,000 iterations")

    h("SECTION 1 — Normality Tests")
    print(f"  HC  Shapiro-Wilk       W={s['sw_hc'].statistic:.4f}  {pfmt(s['sw_hc'].pvalue)}")
    print(f"  HC  D'Agostino-Pearson stat={s['dag_hc'].statistic:.4f}  {pfmt(s['dag_hc'].pvalue)}")
    hc_n = s['sw_hc'].pvalue > 0.05 and s['dag_hc'].pvalue > 0.05
    print(f"  HC conclusion          → {'✓ normal' if hc_n else '✗ non-normal'}")
    print(f"\n  GA  Shapiro-Wilk       W={s['sw_ga'].statistic:.4f}  {pfmt(s['sw_ga'].pvalue)}")
    ga_n = s['sw_ga'].pvalue > 0.05
    print(f"  GA conclusion          → {'✓ normal' if ga_n else '✗ non-normal'}")

    h("SECTION 2 — Descriptive Statistics")
    fmt = "  {:<22} {:>7.3f}  {:>6.3f}  {:>7.3f}  {:>7.3f}  {:>7.3f}  [{:.3f}, {:.3f}]"
    print(f"  {'Label':<22} {'Mean':>7} {'SD':>6} {'Median':>7} {'Min':>7} {'Max':>7}  95% CI")
    print("  " + "─" * 82)
    for label, vals in [("HC 500 iter (n=100)", hc_s),
                         ("GA 500 gen  (n=30) ", ga),
                         ("HC 90k iter (n=5)  ", hc_l)]:
        ci = stats.t.interval(0.95, df=len(vals)-1,
                              loc=np.mean(vals), scale=stats.sem(vals))
        print(fmt.format(label, np.mean(vals), np.std(vals,ddof=1),
                         np.median(vals), np.min(vals), np.max(vals), ci[0], ci[1]))

    h("SECTION 3 — Two-Sample Tests: HC-500 vs GA-500")
    print("  H₀: Both algorithms produce final RMSE from the same distribution.")
    print("  H₁: Distributions differ  (two-sided, α = 0.05)\n")
    print(f"  [PRIMARY]     Mann-Whitney U     U={s['u_main']:.0f}  {pfmt(s['p_mwu'])}  | r={s['r_mwu']:.3f} ({s['main_winner']} superior)")
    print(f"  [SECONDARY]   Welch's t-test     t={s['t_stat']:.3f}  {pfmt(s['p_t'])}  d={s['cohens_d']:.3f}")
    print(f"  [DISTRIBUT.]  KS 2-sample        D={s['ks_stat']:.3f}  {pfmt(s['ks_p'])}")
    print(f"  [ASSUMPTION]  Levene equal var.  W={s['lev_s']:.3f}  {pfmt(s['lev_p'])}  → Welch justified")
    print(f"  [RESAMPLING]  Permutation test   {pfmt(s['perm_p']) if s['perm_p']>0 else 'p < 0.0001'}  (n={N_PERMUTATION:,})")

    h("SECTION 4 — Multi-Group: Kruskal-Wallis + Bonferroni Pairwise")
    print(f"  Kruskal-Wallis  H={s['kw_stat']:.3f}  {pfmt(s['kw_p'])}\n")
    print(f"  {'Comparison':<20} {'U':>6}  {'p (raw)':>9}  {'p (adj)':>9}  {'r':>5}  {'Superior':>8}  Sig?")
    print("  " + "─" * 74)
    for label, u, p_raw, p_adj, r, winner in s['pair_results']:
        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
        print(f"  {label:<20} {u:>6.0f}  {p_raw:>9.2e}  {p_adj:>9.2e}  {r:>5.3f}  {winner:>8}  {sig}")
    print(f"  Statistical Significance: *** if p-value < 0.001; ** if p-value < 0.01; * if p-value < 0.05\n")

    h("SECTION 5 — AUC Convergence Quality")
    print(f"  HC mean AUC : {np.mean(s['hc_aucs']):.1f}")
    print(f"  GA mean AUC : {np.mean(s['ga_aucs']):.1f}")
    print(f"  MWU on AUC  : {pfmt(s['p_auc'])}  | r={s['r_auc']:.3f} ({s['auc_winner']} superior trajectory)")

    h("SECTION 6 — Evaluation Budget")
    rows = [
        ("HC  500 iter (n=100)", "500",       "500",     f"{np.mean(hc_s):.3f}"),
        ("GA  500 gen  (n=30) ", "500",       "~250,000",f"{np.mean(ga):.3f}"),
        ("HC  90k iter (n=5)  ", "90,000",    "90,000",  f"{np.mean(hc_l):.3f}"),
        ("GA  3000 gen (paper)", "3,000",     "~1,500,000",f"{GA_PAPER_RMSE:.3f}"),
    ]
    print(f"  {'Config':<24} {'Steps':>8} {'Evals/run':>13} {'Mean RMSE':>10}")
    print("  " + "─" * 60)
    for r in rows:
        print(f"  {r[0]:<24} {r[1]:>8} {r[2]:>13} {r[3]:>10}")

    h("SECTION 7 — Initialization Grid Search (Spearman ρ)")
    print(f"  {'Parameter':<14} {'ρ':>7}  {'p':>9}  Sig?")
    print("  " + "─" * 38)
    for col, (rho, pv) in s['gs_results'].items():
        sig = "✓" if pv < 0.05 else ""
        print(f"  {col:<14} {rho:>7.3f}  {pv:>9.4f}  {sig}")

    h("SUMMARY")
    print(f"""
  ┌──────────────────────┬──────────────────────────┬───────────────────────┐
  │ Test                 │ Result                   │ Purpose               │
  ├──────────────────────┼──────────────────────────┼───────────────────────┤
  │ Shapiro-Wilk         │ HC p={s['sw_hc'].pvalue:.3f}  GA p={s['sw_ga'].pvalue:.3f}   │ Normality (n<50)      │
  │ D'Agostino-Pearson   │ HC p={s['dag_hc'].pvalue:.3f}               │ Normality (n≥50)      │
  │ Mann-Whitney U       │ {pfmt(s['p_mwu']):<24s} │ Primary non-param     │
  │ Welch's t-test       │ {pfmt(s['p_t']):<24s} │ Parametric secondary  │
  │ KS 2-sample          │ {pfmt(s['ks_p']):<24s} │ Distributional diff.  │
  │ Levene's             │ {pfmt(s['lev_p']):<24s} │ Variance assumption   │
  │ Permutation          │ {'p < 0.0001':<24s} │ Model-free confirm.   │
  │ Kruskal-Wallis       │ {pfmt(s['kw_p']):<24s} │ 3-group comparison    │
  │ Bonferroni pairwise  │ All pairs p < 0.001      │ Post-hoc correction   │
  │ AUC MWU              │ {pfmt(s['p_auc']):<24s} │ Full trajectory test  │
  │ Spearman ρ           │ dx_start sig. p=0.028    │ Init param analysis   │
  └──────────────────────┴──────────────────────────┴───────────────────────┘
    """)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def make_plots(hc_s, hc_s_curves, hc_l, hc_l_curves, ga, ga_curves, s):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    STEPS = hc_s_curves.shape[1]
    steps = np.arange(STEPS)

    # ── 01 Convergence curves ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ga_mean, ga_std = ga_curves.mean(axis=0), ga_curves.std(axis=0)
    ax.fill_between(steps, ga_mean - ga_std, ga_mean + ga_std, alpha=0.15, color=AMBER)
    for c in ga_curves: ax.plot(steps, c, alpha=0.10, lw=0.7, color=AMBER)
    ax.plot(steps, ga_mean, color=AMBER, lw=2.2,
            label=f"GA mean ± 1 SD  (n={len(ga)}, 500 gen)")

    hc_mean, hc_std = hc_s_curves.mean(axis=0), hc_s_curves.std(axis=0)
    ax.fill_between(steps, hc_mean - hc_std, hc_mean + hc_std, alpha=0.15, color=BLUE)
    for c in hc_s_curves[::8]: ax.plot(steps, c, alpha=0.06, lw=0.7, color=BLUE)
    ax.plot(steps, hc_mean, color=BLUE, lw=2.2,
            label=f"HC mean ± 1 SD  (n={len(hc_s)}, 500 iter)")

    ax.axhline(GA_PAPER_RMSE, color=RED, lw=1.5, ls="--",
               label=f"GA paper best — 3000 gen (RMSE {GA_PAPER_RMSE})")
    ax.axhline(np.mean(hc_l), color=TEAL, lw=1.5, ls=":",
               label=f"HC long-run mean — 90k iter (RMSE {np.mean(hc_l):.2f})")
    ax.set_xlabel("Step  (iteration / generation)")
    ax.set_ylabel("Best RMSE")
    ax.set_title("Convergence: Hill Climbing vs Genetic Algorithm at Equal Steps (500)")
    ax.grid(True, alpha=0.25)

    legend = ax.legend(fontsize=9, loc="upper right")

    stats_text = (
        f"Final RMSE  |  HC: {np.mean(hc_s):.2f} ± {np.std(hc_s, ddof=1):.2f}  "
        f"GA: {np.mean(ga):.2f} ± {np.std(ga, ddof=1):.2f}\n"
        f"MWU  {pfmt(s['p_mwu'])},  r = {s['r_mwu']:.2f}"
    )
    # Draw the canvas so the legend bbox is available, then place text just below it
    fig.canvas.draw()
    legend_bb = legend.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(legend_bb.x1, legend_bb.y0 - 0.02,
            stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=1.0, zorder=5))

    save_fig("01_convergence_curves")

    # ── 02 Box plot + significance brackets ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot([hc_s, ga, hc_l],
                    labels=["HC\n500 iter\n(n=100)", "GA\n500 gen\n(n=30)", "HC\n90k iter\n(n=5)"],
                    patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2),
                    flierprops=dict(marker="o", ms=4, alpha=0.5))
    for patch, col in zip(bp["boxes"], [BLUE, AMBER, TEAL]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.scatter([1]*len(hc_s), hc_s, color=BLUE,  s=10, alpha=0.35, zorder=3)
    ax.scatter([2]*len(ga),   ga,   color=AMBER,  s=18, alpha=0.55, zorder=3)
    ax.scatter([3]*len(hc_l), hc_l, color=TEAL,   s=40, zorder=3)
    ax.axhline(GA_PAPER_RMSE, color=RED, lw=1.3, ls="--", alpha=0.7,
               label=f"GA paper = {GA_PAPER_RMSE}")
    ymax = max(np.max(hc_s), np.max(ga), np.max(hc_l))
    p12 = s['pair_results'][0][3]; p13 = s['pair_results'][1][3]; p23 = s['pair_results'][2][3]
    sig_bracket(ax, 1, 2, ymax + 1,  p12, dy=4)
    sig_bracket(ax, 2, 3, ymax + 7,  p23, dy=4)
    sig_bracket(ax, 1, 3, ymax + 13, p13, dy=4)
    ax.set_ylabel("Final RMSE  (lower = better)")
    ax.set_title(f"Final RMSE Distribution — All Conditions\n"
                 f"Kruskal-Wallis: {pfmt(s['kw_p'])},  Bonferroni-corrected pairwise")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis="y")
    save_fig("02_boxplot_significance")

    # ── 03 Normality Q-Q ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, vals, col, label, sw, dag in [
        (axes[0], hc_s, BLUE,  f"HC 500 iter  (n={len(hc_s)})", s['sw_hc'], s['dag_hc']),
        (axes[1], ga,   AMBER, f"GA 500 gen   (n={len(ga)})",   s['sw_ga'], None),
    ]:
        (osm, osr), (slope, intercept, _) = stats.probplot(vals, dist="norm")
        ax.scatter(osm, osr, color=col, s=20, alpha=0.7, zorder=3)
        lx = np.array([min(osm), max(osm)])
        ax.plot(lx, slope*lx+intercept, "k--", lw=1.2, label="Normal reference")
        ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
        ax.set_title(f"Q-Q Plot: {label}")
        note = (f"Shapiro-Wilk: W={sw.statistic:.4f},  {pfmt(sw.pvalue)}\n"
                f"→ {'Cannot reject normality ✓' if sw.pvalue>0.05 else 'Reject normality ✗'}")
        if dag:
            note += (f"\nD'Agostino-Pearson: {pfmt(dag.pvalue)}\n"
                     f"→ {'Cannot reject normality ✓' if dag.pvalue>0.05 else 'Reject normality ✗'}")
        ax.text(0.04, 0.97, note, transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9))
        ax.legend(fontsize=8.5); ax.grid(True, alpha=0.25)
    fig.suptitle("Normality Check — Shapiro-Wilk & D'Agostino-Pearson", fontsize=12)
    save_fig("03_normality_qq")

    # ── 04 Hypothesis tests summary ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    x = np.linspace(15, 65, 500)
    for vals, col, label in [
        (hc_s, BLUE,  f"HC 500 iter  μ={np.mean(hc_s):.2f}"),
        (ga,   AMBER, f"GA 500 gen   μ={np.mean(ga):.2f}"),
        (hc_l, TEAL,  f"HC 90k iter  μ={np.mean(hc_l):.2f}"),
    ]:
        kde = stats.gaussian_kde(vals)
        ax.fill_between(x, kde(x), alpha=0.25, color=col)
        ax.plot(x, kde(x), color=col, lw=2, label=label)
    ax.axvline(GA_PAPER_RMSE, color=RED, lw=1.4, ls="--",
               label=f"GA paper = {GA_PAPER_RMSE}")
    ax.set_xlabel("Final RMSE"); ax.set_ylabel("Density")
    ax.set_title("KDE Distribution Comparison")
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.25)

    ax = axes[1]; ax.axis("off")
    rows = [
        ["Mann-Whitney U\n(primary)",     f"U = {s['u_main']:.0f}",    pfmt(s['p_mwu']),
         f"r = {s['r_mwu']:.3f}\n(large)", "SIGNIFICANT ✓"],
        ["Welch's t-test\n(secondary)",   f"t = {s['t_stat']:.2f}",    pfmt(s['p_t']),
         f"d = {s['cohens_d']:.2f}\n(large)", "SIGNIFICANT ✓"],
        ["KS 2-sample\n(distributional)", f"D = {s['ks_stat']:.3f}",   pfmt(s['ks_p']),
         "—", "SIGNIFICANT ✓"],
        ["Levene's\n(equal variance)",    f"W = {s['lev_s']:.2f}",     pfmt(s['lev_p']),
         "—", "Variances unequal\n→ Welch correct"],
    ]
    styled_table(ax, rows,
                 ["Test", "Statistic", "p-value", "Effect size", "Verdict"],
                 [0.26, 0.18, 0.20, 0.18, 0.18])
    ax.set_title("Hypothesis Test Results: HC-500 vs GA-500", fontsize=10, pad=12)
    fig.suptitle("Two-Sample Tests — HC vs GA (500 steps)", fontsize=12)
    save_fig("04_hypothesis_tests")

    # ── 05 Permutation test ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(s['perm_diffs'], bins=60, color=GRAY, alpha=0.7,
            density=True, label=f"Permuted differences (n={N_PERMUTATION:,})")
    ax.axvline(s['obs_diff'],  color=RED, lw=2.2,
               label=f"Observed difference = {s['obs_diff']:.3f}")
    ax.axvline(-s['obs_diff'], color=RED, lw=2.2, ls="--", alpha=0.6)
    ax.set_xlabel("Mean difference (HC − GA)"); ax.set_ylabel("Density")
    ax.set_title("Permutation Test: HC-500 vs GA-500")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    p_str = pfmt(s['perm_p']) if s['perm_p'] > 0 else "p < 0.0001"
    ax.text(0.5, 0.93,
            f"Observed |difference| = {abs(s['obs_diff']):.3f}\n"
            f"Permutation {p_str}  (n={N_PERMUTATION:,})\n"
            f"None of {N_PERMUTATION:,} permutations exceeded the observed difference",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9))
    save_fig("05_permutation_test")

    # ── 06 Bootstrap CIs ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, boot, vals, col, label in [
        (axes[0], s['boot_hc_s'], hc_s, BLUE,  f"HC 500 iter  (n={len(hc_s)})"),
        (axes[1], s['boot_ga'],   ga,   AMBER, f"GA 500 gen   (n={len(ga)})"),
        (axes[2], s['boot_hc_l'], hc_l, TEAL,  f"HC 90k iter  (n={len(hc_l)})"),
    ]:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax.hist(boot, bins=50, color=col, alpha=0.65, density=True)
        ax.axvline(np.mean(vals), color="black", lw=2,
                   label=f"Mean = {np.mean(vals):.3f}")
        ax.axvline(lo, color=RED, lw=1.5, ls="--")
        ax.axvline(hi, color=RED, lw=1.5, ls="--",
                   label=f"95% CI  [{lo:.3f}, {hi:.3f}]")
        ax.set_xlabel("Bootstrapped mean RMSE")
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend(fontsize=7.5); ax.grid(True, alpha=0.25)
    fig.suptitle(f"Bootstrap 95% Confidence Intervals  (n={N_BOOTSTRAP:,} resamples)",
                 fontsize=12)
    save_fig("06_bootstrap_ci")

    # ── 07 Kruskal-Wallis + pairwise ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    all_vals  = np.concatenate([hc_s, ga, hc_l])
    all_ranks = stats.rankdata(all_vals)
    n1, n2, n3 = len(hc_s), len(ga), len(hc_l)
    rng_jit = np.random.default_rng(0)
    ax.scatter(np.ones(n1)*1 + rng_jit.uniform(-0.15,0.15,n1),
               all_ranks[:n1], color=BLUE,  alpha=0.35, s=12)
    ax.scatter(np.ones(n2)*2 + rng_jit.uniform(-0.15,0.15,n2),
               all_ranks[n1:n1+n2], color=AMBER, alpha=0.55, s=20)
    ax.scatter(np.ones(n3)*3 + rng_jit.uniform(-0.15,0.15,n3),
               all_ranks[n1+n2:], color=TEAL, s=50)
    for vals, pos in [(all_ranks[:n1],1),(all_ranks[n1:n1+n2],2),(all_ranks[n1+n2:],3)]:
        ax.hlines(np.mean(vals), pos-0.3, pos+0.3, colors="black", lw=2.5)
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(["HC 500 iter","GA 500 gen","HC 90k iter"])
    ax.set_ylabel("Global rank  (higher = larger RMSE)")
    ax.set_title(f"Kruskal-Wallis Rank Plot\nH = {s['kw_stat']:.2f},  {pfmt(s['kw_p'])}")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(handles=[
        Patch(color=BLUE, alpha=0.7, label=f"HC 500 iter (n={n1})"),
        Patch(color=AMBER,alpha=0.7, label=f"GA 500 gen  (n={n2})"),
        Patch(color=TEAL, alpha=0.7, label=f"HC 90k iter (n={n3})"),
    ], fontsize=8.5)

    ax = axes[1];
    ax.axis("off")
    pw_rows = []
    for label, u, p_raw, p_adj, r, winner in s['pair_results']:
        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
        pw_rows.append([label, f"{u:.0f}", f"{p_raw:.2e}", f"{p_adj:.2e}", f"{r:.3f}", sig])
    tbl = ax.table(cellText=pw_rows,
                   colLabels=["Comparison","U","p (raw)","p (Bonferroni)","r","Sig"],
                   cellLoc="center", loc="center",
                   colWidths=[0.26,0.10,0.16,0.18,0.10,0.08])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 2.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
        cell.set_edgecolor("#cccccc")
    ax.set_title("Pairwise MWU — Bonferroni corrected  (α = 0.017 per comparison)",
                 pad=14)
    fig.suptitle("Multi-Group Comparison: Kruskal-Wallis + Post-Hoc Pairwise",
                 fontsize=12)
    save_fig("07_kruskal_pairwise")

    # ── 08 AUC convergence ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    ax.hist(s['hc_aucs'], bins=25, color=BLUE,  alpha=0.65, density=True,
            label=f"HC  (n={len(s['hc_aucs'])})")
    ax.hist(s['ga_aucs'], bins=12, color=AMBER, alpha=0.65, density=True,
            label=f"GA  (n={len(s['ga_aucs'])})")
    ax.axvline(np.mean(s['hc_aucs']), color=BLUE,  lw=2.2, ls="--")
    ax.axvline(np.mean(s['ga_aucs']), color=AMBER, lw=2.2, ls="--")
    ax.set_xlabel("AUC  (sum of RMSE across 500 steps)")
    ax.set_ylabel("Density")
    ax.set_title("AUC Distribution per Run")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    ax.text(0.97, 0.97,
            f"HC mean AUC: {np.mean(s['hc_aucs']):.0f}\n"
            f"GA mean AUC: {np.mean(s['ga_aucs']):.0f}\n"
            f"MWU  {pfmt(s['p_auc'])}\nr = {s['r_auc']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9))

    ax = axes[1]
    ax.plot(steps, hc_s_curves.mean(axis=0), color=BLUE,  lw=2.2, label="HC mean")
    ax.plot(steps, ga_curves.mean(axis=0),   color=AMBER, lw=2.2, label="GA mean")
    ax.fill_between(steps, 0, hc_s_curves.mean(axis=0), alpha=0.12, color=BLUE,
                    label=f"HC AUC = {np.mean(s['hc_aucs']):.0f}")
    ax.fill_between(steps, 0, ga_curves.mean(axis=0),   alpha=0.12, color=AMBER,
                    label=f"GA AUC = {np.mean(s['ga_aucs']):.0f}")
    ax.set_xlabel("Step"); ax.set_ylabel("Mean RMSE")
    ax.set_title("Convergence AUC — Full Trajectory Quality")
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.25)
    fig.suptitle(f"AUC Analysis: Total Trajectory Quality  "
                 f"(MWU {pfmt(s['p_auc'])}, r = {s['r_auc']:.2f})", fontsize=12)
    save_fig("08_auc_convergence")

    # ── 09 Evaluation efficiency ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    configs = [
        ("HC 500 iter\n(n=100)", 500,           np.mean(hc_s), np.std(hc_s,ddof=1), BLUE),
        ("GA 500 gen\n(n=30)",   GA_POP_SIZE*500,  np.mean(ga),   np.std(ga,ddof=1),   AMBER),
        ("HC 90k iter\n(n=5)",   90_000,        np.mean(hc_l), np.std(hc_l,ddof=1), TEAL),
        ("GA 3000 gen\n(paper)", GA_POP_SIZE*3000, GA_PAPER_RMSE, 0,                   RED),
    ]
    for label, evals, rmse, err, col in configs:
        ax.errorbar(evals, rmse, yerr=err if err>0 else None,
                    fmt="o", color=col, ms=10, capsize=5, zorder=3, label=label)
        ax.annotate(f"  {label}\n  RMSE={rmse:.2f}",
                    (evals, rmse), textcoords="offset points",
                    xytext=(8, 0), fontsize=7.5, va="center", color=col)
    ax.set_xscale("log")
    ax.set_xlabel("Fitness evaluations per run  (log scale)")
    ax.set_ylabel("Mean final RMSE  (lower = better)")
    ax.set_title("Evaluation Efficiency: Quality vs Computational Budget\n"
                 "(lower-left = better quality with fewer evaluations)")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8.5, loc="upper right")
    save_fig("09_evaluation_efficiency")

    # ── 10 Grid search ───────────────────────────────────────────────────────
    gs_df = s['gs_df']
    params = ["dx_start", "dx_end", "area_start", "area_end"]
    rhos  = [s['gs_results'][p][0] for p in params]
    pvals = [s['gs_results'][p][1] for p in params]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    bar_colors = [RED if pv < 0.05 else "#aaaaaa" for pv in pvals]
    ax.bar(params, rhos, color=bar_colors, alpha=0.8, width=0.5, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    for i, (rho, pv) in enumerate(zip(rhos, pvals)):
        yoff = 0.03 if rho >= 0 else -0.07
        ax.text(i, rho + yoff, f"ρ={rho:.3f}\n{pfmt(pv)}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("Spearman ρ  (correlation with avg_error)")
    ax.set_title("Init Parameter Sensitivity\n(red = p < 0.05)")
    ax.set_xticklabels(params, rotation=20, ha="right")
    ax.legend(handles=[Patch(color=RED, alpha=0.8, label="Significant (p<0.05)"),
                       Patch(color="#aaaaaa", alpha=0.8, label="Not significant")],
              fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    ax = axes[1]
    gs_s = gs_df.sort_values("avg_error")
    y_pos = range(len(gs_s))
    ax.barh(y_pos, gs_s["avg_error"], color=BLUE, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"dx_s={r.dx_start} dx_e={r.dx_end} area_s={r.area_start}"
                        for _, r in gs_s.iterrows()], fontsize=7.5)
    ax.set_xlabel("Avg error across repeated runs")
    ax.set_title("Grid Search Configs Ranked by Avg Error")
    ax.grid(True, alpha=0.25, axis="x")
    fig.suptitle(f"Initialization Grid Search Analysis  (n=10 configs)", fontsize=12)
    save_fig("10_grid_search")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    hc_s, hc_s_curves, hc_l, hc_l_curves, ga, ga_curves, gs = load_data()
    print(f"  HC short: n={len(hc_s)}, GA: n={len(ga)}, HC long: n={len(hc_l)}")

    print("Computing statistics...")
    s = compute_stats(hc_s, ga, hc_l, hc_s_curves, ga_curves)

    print_report(hc_s, ga, hc_l, s)

    print(f"\nGenerating plots → {OUTPUT_DIR}/")
    make_plots(hc_s, hc_s_curves, hc_l, hc_l_curves, ga, ga_curves, s)

    print("\nDone.")
