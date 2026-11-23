import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.utils import load_json
from Q1.q1a_train import HERE

FIG = os.path.join(HERE, "figures")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LABELS = {"mlp": "MLP (1.15M)", "cnn": "CNN (1.15M)", "mlp_wide": "Wide MLP (4.88M)",
          "mlp_aug": "MLP + flip/crop (1.15M)"}

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "legend.frameon": False, "lines.linewidth": 1.8})


def q1a_curves():
    summ = load_json(os.path.join(HERE, "results", "q1a_summary.json"))
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.4), sharey=True)
    for i, name in enumerate(["mlp", "mlp_wide", "mlp_aug", "cnn"]):
        if name not in summ:
            continue
        lr = summ[name]["selected_lr"]
        h = load_json(os.path.join(HERE, "results", f"q1a_{name}_lr{lr}.json"))["history"]
        c = COLORS[i]
        axes[0].plot(h["epoch"], h["train_acc"], color=c, ls="--", label=LABELS[name])
        axes[1].plot(h["epoch"], h["val_acc"], color=c, label=LABELS[name])
    axes[0].set_title("Train accuracy (un-augmented train set)")
    axes[1].set_title("Validation accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylim(0.3, 1.01)
    axes[0].set_ylabel("accuracy")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, "q1a_curves.pdf"))
    fig.savefig(os.path.join(FIG, "q1a_curves.png"), dpi=200)


def crossover_epoch(h):
    for e, f, t in zip(h["epoch"], h["noisy_fit"], h["noisy_true"]):
        if f > t:
            return e
    return None


def q1b_figure_and_table():
    runs = {}
    for nz in [0, 20, 40]:
        p = os.path.join(HERE, "results", f"q1b_noise{nz}.json")
        if os.path.exists(p):
            runs[nz] = load_json(p)

    r = runs[20]
    h = r["history"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
    ax = axes[0]
    ax.plot(h["epoch"], h["clean_fit"], color=COLORS[0], label="train, correct labels")
    ax.plot(h["epoch"], h["noisy_true"], color=COLORS[2], label="relabelled: true class")
    ax.plot(h["epoch"], h["noisy_fit"], color=COLORS[1], label="relabelled: wrong label")
    ax.plot(h["epoch"], h["val_acc"], color=COLORS[3], ls="--", label="validation")
    ax.axvline(r["best_val_epoch"], color="#888888", lw=0.8, ls=":")
    ax.text(r["best_val_epoch"] + 1, 0.04, f"best val (epoch {r['best_val_epoch']:g})", fontsize=7,
            color="#52514e")
    ax.set_title("20% of training labels replaced")
    ax.set_xlabel("epoch")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7)

    ax = axes[1]
    for i, nz in enumerate(sorted(runs)):
        hh = runs[nz]["history"]
        ax.plot(hh["epoch"], hh["val_acc"], color=COLORS[i], label=f"val ({nz}%)")
        if "noisy_fit" in hh:
            ax.plot(hh["epoch"], hh["noisy_fit"], color=COLORS[i], ls="--",
                    label=f"wrong labels fitted ({nz}%)")
    ax.set_title("Effect of the noise level")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q1b_memorisation.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "q1b_memorisation.png"), dpi=200, bbox_inches="tight")

    eps = sorted(set([0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 60.0, r["best_val_epoch"]]))
    rows = []
    for e in eps:
        if e not in h["epoch"]:
            continue
        i = h["epoch"].index(e)
        rows.append([e, h["clean_fit"][i], h["noisy_true"][i], h["noisy_fit"][i], h["val_acc"][i]])
    with open(os.path.join(HERE, "results", "q1b_table_noise20.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "clean_fit", "noisy_true", "noisy_fit", "val_acc"])
        for row in rows:
            w.writerow([f"{row[0]:g}"] + [f"{v:.4f}" for v in row[1:]])

    print("epoch  clean_fit  noisy_true  noisy_fit  val_acc")
    for row in rows:
        print(f"{row[0]:6.3f}  {row[1]:.3f}      {row[2]:.3f}       {row[3]:.3f}      {row[4]:.3f}")
    for nz, rr in sorted(runs.items()):
        hh = rr["history"]
        extra = f"  crossover ep {crossover_epoch(hh)}" if "noisy_fit" in hh else ""
        print(f"noise {nz}%: best val {rr['best_val_acc']:.4f} @ep {rr['best_val_epoch']}  "
              f"final val {hh['val_acc'][-1]:.4f}  test(best-val ckpt) {rr['test_acc_best_val_ckpt']:.4f}  "
              f"test(final) {rr['test_acc_final']:.4f}{extra}")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    q1a_curves()
    q1b_figure_and_table()

