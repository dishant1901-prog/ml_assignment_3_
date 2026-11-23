import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.utils import load_json

SEED = 242003

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "legend.frameon": False})


def load_ae(tag, d):
    import torch
    from Q3.models import ConvAE
    ae = ConvAE(d)
    ae.load_state_dict(torch.load(os.path.join(HERE, "checkpoints", f"q3_{tag}.pt"), map_location="cpu"))
    return ae.eval()


def recon_grid(noise, d_sel):
    import torch
    from common.data import load_cifar, load_split
    from Q3.q3_train_ae import add_noise
    x_all, _, x_test, _ = load_cifar()
    tr, _ = load_split()
    s1, s2, n = np.zeros(3), np.zeros(3), 0
    for i in range(0, len(tr), 5000):
        xb = x_all[tr[i:i + 5000]].astype(np.float64) / 255.0
        s1 += xb.sum(axis=(0, 2, 3))
        s2 += (xb ** 2).sum(axis=(0, 2, 3))
        n += xb.shape[0] * 32 * 32
    m = s1 / n
    mean = torch.tensor(m, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(np.sqrt(s2 / n - m ** 2), dtype=torch.float32).view(1, 3, 1, 1)
    del x_all
    idx = np.random.RandomState(SEED).choice(len(x_test), 8, replace=False)
    x = (torch.tensor(x_test[idx], dtype=torch.float32) / 255.0 - mean) / std
    xn = add_noise(x, noise, mean, std, torch.Generator().manual_seed(SEED + 3))
    rows = [("original", x)]
    with torch.no_grad():
        for d in (16, 128, 1024):
            rows.append((f"AE-{d}", load_ae(f"ae_d{d}", d)(x)))
        rows.append((f"noisy (std {noise})", xn))
        rows.append((f"AE-{d_sel}(noisy)", load_ae(f"ae_d{d_sel}", d_sel)(xn)))
        rows.append((f"DAE-{d_sel}(noisy)", load_ae(f"ae_d{d_sel}_noise{noise}", d_sel)(xn)))
    fig, axes = plt.subplots(len(rows), 8, figsize=(3.3, 3.0))
    for r, (name, imgs) in enumerate(rows):
        pix = (imgs * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
        for c in range(8):
            ax = axes[r, c]
            ax.imshow(pix[c])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=6)
    fig.subplots_adjust(left=0.27, right=0.99, top=0.99, bottom=0.01, wspace=0.05, hspace=0.05)
    fig.savefig(os.path.join(FIG, "q3_reconstructions.png"), dpi=250)
    fig.savefig(os.path.join(FIG, "q3_reconstructions.pdf"))


def recon_vs_class_info(summ):
    c = summ["c"]
    names = [k for k in c if k != "raw pixels"]
    offsets = {"AE-16": (-8, 6), "AE-128": (6, 4), "AE-1024": (-12, -14), "DAE-1024": (6, 2)}
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for i, n in enumerate(names):
        x = c[n]["test_mse_pixel"]
        lp, kn = c[n]["linear_probe"]["test_acc"], c[n]["knn"]["test_acc"]
        ax.plot([x, x], [lp, kn], color=COLORS[i % 4], lw=1, zorder=2)
        ax.scatter(x, lp, s=30, color=COLORS[i % 4], edgecolor="white", linewidth=1.5, zorder=3,
                   label="linear probe" if i == 0 else None)
        ax.scatter(x, kn, s=26, marker="s", facecolor="white", edgecolor=COLORS[i % 4], linewidth=1.5, zorder=3,
                   label="k-NN" if i == 0 else None)
        ax.annotate(n, (x, kn if n == "AE-1024" else lp), fontsize=6.5, xytext=offsets.get(n, (5, 3)),
                    textcoords="offset points")
    raw = c["raw pixels"]
    ax.axhline(raw["linear_probe"]["test_acc"], color="#888888", lw=0.8, ls=":")
    ax.text(0.0021, raw["linear_probe"]["test_acc"] - 0.018, "raw pixels, linear (0.415)", fontsize=6,
            color="#52514e")
    ax.set_xscale("log")
    ax.set_xlabel("test reconstruction MSE ([0,1] pixels, log scale)")
    ax.set_ylabel("test accuracy")
    ax.legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q3_recon_vs_probe.png"), dpi=250)
    fig.savefig(os.path.join(FIG, "q3_recon_vs_probe.pdf"))


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    s = load_json(os.path.join(HERE, "results", "q3_summary.json"))
    if "--scatter-only" not in sys.argv:
        recon_grid(s["noise"], s["a"]["selected"])
    recon_vs_class_info(s)

