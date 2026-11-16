"""README charts. Series colours: blue (model), orange (LACE), grey (random)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a94a0"
INK, MUTED, GRID = "#1f2a37", "#6b7480", "#e3e6ea"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "text.color": INK,
})


def gains_chart(y, scores: dict, path, mark=0.20):
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    styles = {"Risk model (gradient boosting)": BLUE, "LACE index": ORANGE}
    for name, s in scores.items():
        order = np.argsort(-np.asarray(s), kind="stable")
        caught = np.cumsum(y[order]) / y.sum()
        share = np.arange(1, len(y) + 1) / len(y)
        ax.plot(share * 100, caught * 100, color=styles[name], lw=2, label=name)
        k = int(round(mark * len(y))) - 1
        ax.plot(mark * 100, caught[k] * 100, "o", color=styles[name], ms=7, markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"{caught[k]:.0%}", (mark * 100, caught[k] * 100), xytext=(8, -4),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.plot([0, 100], [0, 100], color=GREY, lw=1.2, ls="--", label="Random selection")
    ax.axvline(mark * 100, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Discharges contacted (%), highest risk first")
    ax.set_ylabel("30-day readmissions among those contacted (%)")
    ax.set_title("Who to call: share of readmissions reached")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path); plt.close(fig)


def impact_chart(impact, path):
    """impact: DataFrame with capacity, method, avoided, low, high (per 1,000 discharges)."""
    caps = sorted(impact["capacity"].unique())
    methods = [("Risk model (gradient boosting)", BLUE), ("LACE index", ORANGE), ("Random selection", GREY)]
    x = np.arange(len(caps)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for i, (m, c) in enumerate(methods):
        d = impact[impact["method"] == m].set_index("capacity").loc[caps]
        ax.bar(x + (i - 1) * w, d["avoided"], w * 0.92, color=c, label=m)
        ax.errorbar(x + (i - 1) * w, d["avoided"], yerr=[d["avoided"] - d["low"], d["high"] - d["avoided"]],
                    fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
    ax.set_xticks(x, [f"{c:.0%}" for c in caps])
    ax.set_xlabel("Follow-up capacity (share of discharges contacted)")
    ax.set_ylabel("Readmissions avoided per 1,000 discharges")
    ax.set_title("Expected impact of a transitional-care programme")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.savefig(path); plt.close(fig)


def calibration_chart(cal_gbm, cal_lr, path):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = max(cal_gbm[["predicted", "observed"]].max().max(), cal_lr[["predicted", "observed"]].max().max()) * 1.08
    ax.plot([0, lim], [0, lim], color=GREY, lw=1, ls="--", label="Perfect calibration")
    ax.plot(cal_gbm["predicted"], cal_gbm["observed"], "o-", color=BLUE, lw=2, ms=6, label="Gradient boosting")
    ax.plot(cal_lr["predicted"], cal_lr["observed"], "s-", color=ORANGE, lw=2, ms=6, label="Logistic regression")
    ax.set_xlabel("Predicted 30-day readmission risk (decile mean)")
    ax.set_ylabel("Observed readmission rate")
    ax.set_title("Calibration by decile")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(path); plt.close(fig)


def drivers_chart(imp, path, top=12):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(d["label"], d["auc_drop"] * 1000, color=BLUE, height=0.62)
    ax.set_xlabel("Drop in AUC when the feature is shuffled (x 1,000)")
    ax.set_title("What drives predicted readmission risk")
    ax.grid(axis="y", visible=False)
    fig.savefig(path); plt.close(fig)
