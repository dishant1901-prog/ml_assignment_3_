import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.utils import load_json

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
C_MLP, C_CNN, C3 = "#2a78d6", "#eb6834", "#1baf7a"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "legend.frameon": False,
                     "axes.axisbelow": True})
COND_LABEL = {"clean": "clean", "perm": "pixel\nperm.", "patch16": "16x16\npatches", "patch8": "8x8\npatches",
              "patch4": "4x4\npatches"}


def accuracy_by_condition(s):
    conds = [c for c in COND_LABEL if f"{c}/cnn" in s or f"{c}/mlp" in s]
    x = np.arange(len(conds))
    w = 0.36
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for off, model, col, lab in ((-w / 2, "mlp", C_MLP, "MLP (1.15M)"), (w / 2, "cnn", C_CNN, "CNN (1.15M)")):
        vals = [s.get(f"{c}/{model}", {}).get("test_acc", np.nan) for c in conds]
        bars = ax.bar(x + off, vals, w - 0.04, color=col, label=lab)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
    ax.set_xticks(x, [COND_LABEL[c] for c in conds], fontsize=6.5)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q2_accuracy_by_condition.png"), dpi=250)
    fig.savefig(os.path.join(FIG, "q2_accuracy_by_condition.pdf"))


def reliability(probs, y, bins=10):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 20:
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
            ns.append(m.sum())
    return np.array(xs), np.array(ys)


def confidence_figure(p):
    y = p["y"]
    runs = [("clean__cnn", "CNN, clean", C_CNN, "-"), ("patch8__cnn", "CNN, 8x8 patches (trained on them)", C3, "-"),
            ("cleantrained_on_patch8__cnn", "clean CNN on 8x8 patches", C_MLP, "--")]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8)
    for key, lab, col, ls in runs:
        if key in p:
            xs, ys = reliability(p[key], y)
            ax.plot(xs, ys, color=col, ls=ls, marker="o", ms=3, lw=1.6, label=lab)
    ax.set_xlabel("confidence (max softmax)")
    ax.set_ylabel("accuracy in bin")
    ax.set_title("Reliability on test (bins with >= 20 images)", fontsize=8)
    ax.legend(loc="upper left", fontsize=6.5)
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    for key, lab, col, ls in runs[:2]:
        if key in p:
            conf = p[key].max(1)
            wrong = p[key].argmax(1) != y
            ax.hist(conf[wrong], bins=bins, histtype="step", color=col, lw=1.6, label=f"{lab}: wrong",
                    density=True)
            ax.hist(conf[~wrong], bins=bins, histtype="step", color=col, lw=1.0, ls=":",
                    label=f"{lab}: correct", density=True)
    ax.set_xlabel("confidence (max softmax)")
    ax.set_ylabel("density")
    ax.set_title("Confidence of right and wrong predictions", fontsize=8)
    ax.legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q2_confidence.png"), dpi=250)
    fig.savefig(os.path.join(FIG, "q2_confidence.pdf"))


def per_class_figure(s):
    if "clean/cnn" not in s or "patch8/cnn" not in s:
        return
    a = np.array(s["clean/cnn"]["per_class_acc"])
    b = np.array(s["patch8/cnn"]["per_class_acc"])
    m = np.array(s["perm/cnn"]["per_class_acc"]) if "perm/cnn" in s else None
    x = np.arange(10)
    w = 0.27
    fig, ax = plt.subplots(figsize=(6.9, 1.9))
    ax.bar(x - w, a, w - 0.03, color=C_CNN, label="CNN clean")
    ax.bar(x, b, w - 0.03, color=C3, label="CNN 8x8 patches")
    if m is not None:
        ax.bar(x + w, m, w - 0.03, color=C_MLP, label="CNN pixel perm.")
    ax.set_xticks(x, CLASSES, fontsize=7)
    ax.set_ylabel("test recall")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", ncol=3, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q2_per_class.png"), dpi=250)
    fig.savefig(os.path.join(FIG, "q2_per_class.pdf"))


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    s = load_json(os.path.join(HERE, "results", "q2_summary.json"))
    p = dict(np.load(os.path.join(HERE, "results", "q2_test_probs.npz")))
    accuracy_by_condition(s)
    confidence_figure(p)
    per_class_figure(s)

