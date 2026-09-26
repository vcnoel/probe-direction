"""The six body figures, drawn from the result files at the manuscript's text width.

Restored from the repository's former scripts/make_figures.py (removed when the manuscript was
untracked) and rebuilt for the multi-ladder design. Every ladder present in the result files is
drawn, newest first, so the Qwen3.5 ladder appears in each figure once scripts/pod_ladders.py and
scripts/ladder_curves.py have been run on its arrays; no edit here is needed.

  fig1_wrappers   the statistic against parameters on every ladder, all 36 wrappers in grey and the
                  two that share one evaluation framing and spread r widest in colour
  fig5_signdist   the 36 correlations with log parameters on every ladder, the two published values
                  placed on the Qwen2.5 row
  fig3_variance   variance shares of the three-way crossed design on every ladder, and the two
                  extended designs on Qwen2.5
  fig4_dstudy     E rho^2 against wrappers averaged and against item sets averaged, every ladder
  fig2_floor      published values of the released study against the floor of its own estimator
  fig6_subsets    agreement of a k-wrapper average with the 36-wrapper average and between
                  disjoint k-subsets, for the scaling sign and the model ordering

Type is the document's own (pgf backend, pdflatex, times). Figures are saved at the text width and
included with no width argument. No number is typed here: each is read from the file the prose
reads.

    python paper_tools/make_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(os.environ.get("EAS_REPO", Path(__file__).resolve().parents[1]))
# Generated .tex and figure files go to paper_tools/out/ unless EAS_OUT names another directory.
HERE = Path(os.environ.get("EAS_OUT", REPO / "paper_tools" / "out"))
HERE.mkdir(parents=True, exist_ok=True)
RES = REPO / "results"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
from gstudy_gtheory import coefficients  # noqa: E402

TEXTWIDTH = 397.485 / 72.27          # inches, \the\textwidth of the ICLR template

INK, MID, FAINT = "#1A1A1A", "#7A7A7A", "0.86"
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
SURFACE = "#ffffff"

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex", "text.usetex": True, "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{times}\usepackage{amsmath}",
    "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "lines.linewidth": 1.1, "legend.frameon": False,
    "savefig.bbox": None,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

# Result-file key -> display name. Order is newest generation first, the order of tab:ladders.
LADDERS = [("qwen3.5_nothink", "Qwen3.5"), ("qwen3.5_think", "Qwen3.5, thinking on"),
           ("qwen3_nothink", "Qwen3"), ("qwen3_think", "Qwen3, thinking on"),
           ("olmo2", "OLMo-2"), ("qwen2.5_paper", "Qwen2.5")]
PRIMARY = ["qwen3.5_nothink", "qwen3_nothink", "olmo2", "qwen2.5_paper"]
# One colour per ladder, the same in every figure. The newest ladder (Qwen3.5, 2026) leads in
# blue, drawn heavier and on top. Qwen2.5 keeps orange because the reanalysed studies and their
# released artifacts are on it. Qwen3 and OLMo-2 are grey context, labelled at their ends.
COLOUR = {"qwen3.5_nothink": BLUE, "qwen3.5_think": BLUE, "qwen3_nothink": MID,
          "qwen3_think": MID, "olmo2": MID, "qwen2.5_paper": ORANGE}
LEAD = {"qwen3.5_nothink", "qwen3.5_think"}
DASHED = {"qwen3.5_think", "qwen3_think"}


def lw(k, base=1.1):
    """Line width: the leading ladder heavier, context ladders lighter."""
    return base * (1.35 if k in LEAD else 0.8 if COLOUR[k] == MID else 1.0)


def z(k):
    return 5 if k in LEAD else 3 if COLOUR[k] != MID else 2


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def tt(s):
    return r"\texttt{" + s + "}"


def title(ax, text):
    ax.set_title(text, loc="left", pad=3)


def repel(ys, gap, floor=-np.inf, ceil=np.inf):
    """Spread label heights so neighbours sit at least gap apart, keeping their order and staying
    between floor and ceil: an upward pass from the lowest, then a downward pass from the highest."""
    order = np.argsort(ys)
    out = np.clip(np.array(ys, float), floor, ceil)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    if out[order[-1]] > ceil:
        out[order[-1]] = ceil
        for a, b in zip(order[::-1][:-1], order[::-1][1:]):
            if out[a] - out[b] < gap:
                out[b] = out[a] - gap
    return out


def present(keys):
    lc = load("ladder_curves.json")["ladders"]
    return [k for k in keys if k in lc], lc


def widest_pair(pairs, r):
    """Within the evaluation framing whose deployment variants spread r widest, the two extremes."""
    by = {}
    for j, (e, _) in enumerate(pairs):
        by.setdefault(e, []).append(j)
    e = max(by, key=lambda f: max(r[j] for j in by[f]) - min(r[j] for j in by[f]))
    js = by[e]
    return max(js, key=lambda j: r[j]), min(js, key=lambda j: r[j])


def fig1_wrappers():
    keys, lc = present(PRIMARY)
    fig, axes = plt.subplots(1, len(keys), figsize=(TEXTWIDTH, 1.8), sharey=True)
    axes = np.atleast_1d(axes)
    allv = np.concatenate([np.ravel(lc[k]["statistic"]) for k in keys])
    lo, hi = np.floor(allv.min() * 50) / 50, np.ceil(allv.max() * 50) / 50
    for i, (ax, k) in enumerate(zip(axes, keys)):
        d = lc[k]
        p, x, r = np.array(d["params_b"]), np.array(d["statistic"]), np.array(d["r"])
        for j in range(x.shape[1]):
            ax.plot(p, x[:, j], color=FAINT, lw=0.6, zorder=1)
        jp, jn = widest_pair(d["pairs"], r)
        ends = []
        for j, col in ((jp, BLUE), (jn, ORANGE)):
            ax.plot(p, x[:, j], "o-", color=col, ms=2.4, lw=1.2, zorder=3)
            ends.append((x[-1, j], col, r[j]))
        ys = repel([e[0] for e in ends], (hi - lo) * 0.11)
        for (y0, col, rv), y in zip(ends, ys):
            ax.text(p[-1] * 1.25, y, f"${rv:+.2f}$", color=col, fontsize=7.0, va="center")
        ax.set_xscale("log")
        ax.set_xlim(p[0] / 1.35, p[-1] * 4.6)
        ticks = [t for t in (0.5, 1, 3, 7, 30) if p[0] / 1.1 <= t <= p[-1] * 1.1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        name = dict(LADDERS)[k]
        title(ax, f"({'abcd'[i]}) " + tt(name))
    axes[0].set_ylim(lo, hi)
    axes[0].set_ylabel(r"$|\mathrm{AUROC} - 0.5|$, best layer")
    fig.supxlabel("parameters (B)", fontsize=8.5, y=0.02)
    fig.subplots_adjust(wspace=0.10, bottom=0.21, top=0.88, left=0.09, right=0.995)
    fig.savefig(FIG / "fig1_wrappers.pdf")
    plt.close(fig)


def fig5_signdist():
    keys, lc = present([k for k, _ in LADDERS])
    pub = load("slope_distribution_deploy_peak_lf.json")["published"]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 0.21 * len(keys) + 0.75))
    ys = np.arange(len(keys))[::-1] + 1.0
    for y, k in zip(ys, keys):
        r = np.sort(np.array(lc[k]["r"]))
        ax.plot([r.min(), r.max()], [y, y], color=FAINT, lw=3.0, solid_capstyle="round",
                zorder=1)
        ax.plot(r, np.full_like(r, y), "|", color=COLOUR[k], ms=7,
                mew=1.1 if k in LEAD else 0.9, zorder=z(k))
    ax.axvline(0, color=MID, lw=0.5, ls=(0, (2, 2)), zorder=0)
    # The two triangles are OUR fits to each study's released Qwen2.5 files, so they sit on that
    # row. Neither is a value either study published: the first study reports a qualitative
    # power-law trend over 15 non-Qwen models and no correlation at all. The labels say so. The axis-ambiguous variant is discussed in the text and not drawn.
    yq = ys[keys.index("qwen2.5_paper")]
    for nm, v in pub.items():
        if "un-transposed" in nm:
            continue
        lab = nm.split(",")[0].replace(" et al.", "")
        what = "files" if lab.startswith("Chaudhary") else "values"
        ax.plot(v, yq - 0.42, "^", color=INK, ms=4.0, zorder=4, clip_on=False)
        ax.text(v, yq - 0.74, f"our fit to {lab} {what}, ${v:+.2f}$", ha="center", va="top",
                fontsize=7.0, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([tt(n.split(",")[0]) + ("" if "," not in n else ", thinking on")
                        for k in keys for kk, n in LADDERS if kk == k])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(yq - 1.35, ys.max() + 0.5)
    ax.set_xlabel(r"correlation of the statistic with $\log_{10}$ parameters, one mark per wrapper")
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.50 / fig.get_figheight(),
                        top=1 - 0.08 / fig.get_figheight())
    fig.savefig(FIG / "fig5_signdist.pdf")
    plt.close(fig)


def fig3_variance():
    gt = load("pod_gtheory.json")["ladders"]
    impl = load("implementation_facet.json")["share"]
    reml = load("reml_components_validated_interior.json")["share"]
    cats = [("model", BLUE), (r"model$\times$wrapper", ORANGE), (r"model$\times$item", MID),
            ("implementation", GREEN), ("all other components", FAINT)]
    rows = []
    for k, nm in LADDERS:
        if k in gt:
            s = gt[k]["shares"]
            rows.append((tt(nm.split(",")[0]) + (", thinking on" if "," in nm else ""),
                         [s["s_m"], s["s_mt"], s["s_mh"], 0.0]))
    rows.append((tt("Qwen2.5") + ", with implementation",
                 [impl["model"], impl["modelxtemplate"], impl["modelxitem"], impl["impl"]]))
    rows.append(("seven models, REML", [reml["model(family)"], reml["model:template"],
                                     reml["model:item"], 0.0]))
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 0.15 * len(rows) + 0.75))
    ys = np.arange(len(rows))[::-1].astype(float)
    ys[-2:] -= 0.5                        # a gap separates the extended designs from the ladders
    for y, (_, v) in zip(ys, rows):
        v = [100 * a for a in v]
        v.append(100 - sum(v))
        assert v[-1] > -0.5, "shares exceed 100"
        left = 0.0
        for a, (_, col) in zip(v, cats):
            ax.barh(y, a, left=left, height=0.66, color=col, edgecolor=SURFACE, lw=0.4, zorder=2)
            left += a
    ax.set_yticks(ys)
    ax.set_yticklabels([nm for nm, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of total variance (\\%)")
    fig.subplots_adjust(left=0.30, right=0.985, bottom=0.42 / fig.get_figheight(),
                        top=1 - 0.30 / fig.get_figheight())
    fig.legend(handles=[Patch(color=c, label=n) for n, c in cats], ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), handlelength=1.0, handleheight=0.8,
              columnspacing=1.0, handletextpad=0.4, borderaxespad=0.2)
    fig.savefig(FIG / "fig3_variance.pdf")
    plt.close(fig)


def fig4_dstudy():
    gt = load("pod_gtheory.json")["ladders"]
    keys = [k for k, _ in LADDERS if k in gt]
    grid = np.unique(np.round(np.logspace(0, 3, 160)).astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.8), sharey=True)
    for i, ax in enumerate(axes):
        ends = []
        for k in keys:
            c, n = gt[k]["components"], gt[k]["n_blocks"]
            # (a) k wrappers on the full item set; (b) one wrapper on k item sets of the full size
            e = [coefficients(c, k=g, n=n)[0] if i == 0 else coefficients(c, k=1, n=n * g)[0]
                 for g in grid]
            ax.plot(grid, e, color=COLOUR[k], lw=lw(k),
                    ls=(0, (3, 1.5)) if k in DASHED else "-", zorder=z(k))
            ends.append((e[-1], k))
        ly = repel([a for a, _ in ends], 0.095, floor=0.08, ceil=0.97)
        for (_, k), y in zip(ends, ly):
            nm = dict(LADDERS)[k].replace(", thinking on", ", think")
            ax.text(grid[-1] * 1.25, y, tt(nm.split(",")[0]) + (", think" if "," in nm else ""),
                    color=COLOUR[k], fontsize=7.0, va="center")
        ax.plot([1, grid[-1]], [0.80, 0.80], color=MID, lw=0.5, ls=(0, (2, 2)), zorder=1)
        ax.text(1.1, 0.835, "0.80", color=MID, fontsize=7.0, va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(1, grid[-1] * 14)
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(["1", "10", "100", "1000"])
        ax.set_ylim(0, 1.02)
    axes[0].set_xlabel("wrappers averaged, full item set")
    axes[1].set_xlabel("item sets averaged, one wrapper")
    axes[0].set_ylabel(r"$E\rho^2$")
    title(axes[0], r"(a) $E\rho^2$ against wrappers")
    title(axes[1], r"(b) $E\rho^2$ against items")
    fig.subplots_adjust(wspace=0.08, bottom=0.21, top=0.88, left=0.095, right=0.99)
    fig.savefig(FIG / "fig4_dstudy.pdf")
    plt.close(fig)


def fig2_floor():
    d = load("released_label_permuted.json")["per_model"]
    keys = sorted(d, key=lambda k: float(k.split("-")[-1].rstrip("b")))
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 1.8))
    y = np.arange(len(keys))[::-1].astype(float)
    for yi, k in zip(y, keys):
        r = d[k]
        fs, fu = r["variants"]["system"], r["variants"]["user"]
        ax.errorbar(fs["perm_mean"], yi + 0.14, xerr=fs["perm_sd"], fmt="o", ms=3.4, color=MID,
                    mfc=MID, elinewidth=0.7, capsize=1.6, zorder=3)
        ax.errorbar(fu["perm_mean"], yi - 0.14, xerr=fu["perm_sd"], fmt="s", ms=3.2, color=MID,
                    mfc=SURFACE, mew=0.8, elinewidth=0.7, capsize=1.6, zorder=3)
        ax.plot(r["published"], yi, "D", ms=4.0, color=BLUE, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([tt("Qwen2.5-" + k.split("-")[-1].upper()) for k in keys])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlabel(r"$|\mathrm{AUROC} - 0.5|$ at the best layer")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="D", ls="", ms=4.0, color=BLUE, label="published value"),
        Line2D([], [], marker="o", ls="", ms=3.4, color=MID, label="floor, system rendering"),
        Line2D([], [], marker="s", ls="", ms=3.2, mfc=SURFACE, mec=MID,
               label="floor, user rendering")],
        loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=1.0, handletextpad=0.4,
        labelspacing=0.5)
    fig.subplots_adjust(left=0.18, right=0.74, bottom=0.22, top=0.97)
    fig.savefig(FIG / "fig2_floor.pdf")
    plt.close(fig)


def fig6_subsets():
    """Agreement of a k-wrapper average with the 36-wrapper average (solid) and between two
    disjoint k-subsets (dotted), for the scaling sign and the model ordering."""
    d = load("wrapper_subsets.json")
    lad = d["ladders"]
    keys = [k for k in PRIMARY if k in lad]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.9))
    spec = [("sign_agreement", "disjoint_sign_agreement", d["sign_target"], (0.3, 1.03),
             r"(a) scaling sign agrees", "share of draws"),
            ("mean_tau", "disjoint_mean_tau", d["tau_target"], (0.0, 1.03),
             r"(b) model ordering agrees", r"mean Kendall $\tau$")]
    for ax, (f, fd, target, ylim, head, ylab) in zip(axes, spec):
        starts = []
        for k in keys:
            r = lad[k]
            ax.plot(r["k"], r[f], color=COLOUR[k], lw=lw(k), zorder=z(k))
            ax.plot(r["disjoint_k"], r[fd], color=COLOUR[k], lw=lw(k, 0.9), ls=(0, (1, 1.3)),
                    zorder=z(k) - 1)
            starts.append((r[f][0], k))
        ly = repel([a for a, _ in starts], 0.085 * (ylim[1] - ylim[0]), floor=ylim[0] + 0.04,
                   ceil=ylim[1] - 0.04)
        for (_, k), y in zip(starts, ly):
            ax.text(0.85, y, tt(dict(LADDERS)[k]), color=COLOUR[k], fontsize=7.0, va="center",
                    ha="right")
        ax.plot([1, 36], [target, target], color=MID, lw=0.5, ls=(0, (2, 2)), zorder=1)
        ax.text(1.05, target + 0.015 * (ylim[1] - ylim[0]), f"{target:.2f}", color=MID,
                fontsize=7.0, va="bottom", ha="left")
        ax.set_xscale("log")
        ax.set_xlim(0.32, 40)
        ax.set_xticks([1, 2, 5, 10, 18, 36])
        ax.set_xticklabels(["1", "2", "5", "10", "18", "36"])
        ax.minorticks_off()
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
        ax.set_xlabel("wrappers averaged, $k$")
        title(ax, head)
    fig.subplots_adjust(wspace=0.30, bottom=0.21, top=0.88, left=0.085, right=0.99)
    fig.savefig(FIG / "fig6_subsets.pdf")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig1_wrappers, fig5_signdist, fig3_variance, fig4_dstudy, fig2_floor,
              fig6_subsets):
        f()
        print(f"  {f.__name__}")
    print(f"wrote {FIG} at {TEXTWIDTH:.3f}in, include with no width argument")


if __name__ == "__main__":
    main()
