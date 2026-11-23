import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
KCOL = {"linear": "#2a78d6", "rbf": "#eb6834", "poly3": "#1baf7a"}
KLAB = {"linear": "linear", "rbf": "RBF", "poly3": "poly (deg 3)"}
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "legend.frameon": False})


def sweep_figure():
    rows = json.load(open(os.path.join(RES, "svm_sweep.json")))
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3))
    for k in KCOL:
        r = sorted([x for x in rows if x["kernel"] == k], key=lambda x: x["C"])
        C = [x["C"] for x in r]
        axes[0].plot(C, [x["val_acc"] for x in r], color=KCOL[k], marker="o", ms=4, lw=1.8,
                     label=f"{KLAB[k]}: val")
        axes[0].plot(C, [x["train_acc"] for x in r], color=KCOL[k], ls="--", lw=1.2, label=f"{KLAB[k]}: train")
        axes[1].plot(C, [x["sv_fraction"] for x in r], color=KCOL[k], marker="o", ms=4, lw=1.8, label=KLAB[k])
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Train (10k subset) and validation accuracy", fontsize=8)
    axes[0].legend(fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    axes[1].set_ylabel("fraction of training points")
    axes[1].set_title("Support-vector fraction", fontsize=8)
    axes[1].legend(fontsize=6.5)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("C (log scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q4_sweep.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "q4_sweep.pdf"), bbox_inches="tight")


def contingency_figure():
    ct = json.load(open(os.path.join(RES, "cluster_test.json")))
    cont = np.array(ct["contingency"])
    order = np.lexsort((-cont.max(0), cont.argmax(0)))
    cont = cont[:, order]
    fig, ax = plt.subplots(figsize=(3.3 if cont.shape[1] <= 12 else 6.9, 2.6))
    im = ax.imshow(cont, cmap="Blues", aspect="auto")
    for i in range(cont.shape[0]):
        for j in range(cont.shape[1]):
            v = cont[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=5,
                    color="white" if v > cont.max() * 0.55 else "#0b0b0b")
    ax.set_yticks(range(10), CLASSES, fontsize=6.5)
    ax.set_xticks(range(cont.shape[1]), [str(o) for o in order], fontsize=6)
    ax.set_xlabel("cluster id")
    ax.grid(False)
    ax.set_title(f"Test: {ct['selected']['method']} k={ct['selected']['k']}, "
                 f"ARI {ct['test_ari']:.3f}, NMI {ct['test_nmi']:.3f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q4_contingency.png"), dpi=250)
    fig.savefig(os.path.join(FIG, "q4_contingency.pdf"))


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    sweep_figure()
    if os.path.exists(os.path.join(RES, "cluster_test.json")):
        contingency_figure()

