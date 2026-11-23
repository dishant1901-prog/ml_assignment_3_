import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from common.data import get_data
from common.models import CNN
from common.train import evaluate
from common.utils import load_json, save_json, set_seed
from Q3.models import ConvAE
from Q3.q3_train_ae import add_noise, recon_error, SEED

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BOTTLENECKS = [16, 128, 1024]


def load_ae(tag, d):
    ae = ConvAE(d).cuda()
    ae.load_state_dict(torch.load(os.path.join(HERE, "checkpoints", f"q3_{tag}.pt")))
    return ae.eval()


@torch.no_grad()
def reconstruct(ae, x, bs=1000):
    return torch.cat([ae(x[i:i + bs]) for i in range(0, len(x), bs)])


@torch.no_grad()
def codes(ae, x, bs=1000):
    return torch.cat([ae.encode(x[i:i + bs]) for i in range(0, len(x), bs)])


def linear_probe(ftr, ytr, fva, yva, fte, yte):
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    ftr, fva, fte = (ftr - mu) / sd, (fva - mu) / sd, (fte - mu) / sd
    best = None
    for wd in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        set_seed(SEED)
        lin = torch.nn.Linear(ftr.shape[1], 10).cuda()
        opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=300, history_size=20,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(lin(ftr), ytr) + wd * (lin.weight ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            va = (lin(fva).argmax(1) == yva).float().mean().item()
            if best is None or va > best[1]:
                te = (lin(fte).argmax(1) == yte).float().mean().item()
                tr = (lin(ftr).argmax(1) == ytr).float().mean().item()
                best = (wd, va, te, tr)
    return {"wd": best[0], "val_acc": best[1], "test_acc": best[2], "train_acc": best[3]}


@torch.no_grad()
def knn(ftr, ytr, fva, yva, fte, yte):
    def pred(f, k):
        out = []
        a = F.normalize(ftr, dim=1)
        for i in range(0, len(f), 1000):
            s = F.normalize(f[i:i + 1000], dim=1) @ a.T
            idx = s.topk(k, dim=1).indices
            out.append(torch.mode(ytr[idx], dim=1).values)
        return torch.cat(out)
    best = None
    for k in [1, 5, 10, 20, 50, 100]:
        va = (pred(fva, k) == yva).float().mean().item()
        if best is None or va > best[1]:
            best = (k, va)
    te = (pred(fte, best[0]) == yte).float().mean().item()
    return {"k": best[0], "val_acc": best[1], "test_acc": te}


def main(noise, parts):
    data = get_data()
    std, mean = data["std"], data["mean"]
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    xte, yte = data["test"]
    path = os.path.join(HERE, "results", "q3_summary.json")
    out = load_json(path) if os.path.exists(path) else {}
    out["noise"] = noise

    val_err = {}
    for d in BOTTLENECKS:
        r = load_json(os.path.join(HERE, "results", f"q3_ae_d{d}.json"))
        val_err[d] = {"params": r["params"], "val_mse": r["history"]["val_mse"][-1],
                      "val_mse_pixel": r["history"]["val_mse_pixel"][-1]}
    d_sel = min(val_err, key=lambda d: val_err[d]["val_mse"])
    ae_sel = load_ae(f"ae_d{d_sel}", d_sel)
    tn, tp = recon_error(ae_sel, xte, xte, std)
    out["a"] = {"by_bottleneck": val_err, "selected": d_sel, "test_mse": tn, "test_mse_pixel": tp,
                "test_psnr": float(10 * np.log10(1 / tp))}
    print("(a)", out["a"])

    if "b" in parts:
        out["b"] = part_b(data, noise, d_sel)
    if "c" in parts:
        out["c"] = part_c(data, noise, d_sel)
    save_json(out, path)


def part_b(data, noise, d_sel):
    std, mean = data["std"], data["mean"]
    xte, yte = data["test"]
    ae_sel = load_ae(f"ae_d{d_sel}", d_sel)
    q1 = load_json(os.path.join(ROOT, "Q1", "results", "q1a_summary.json"))["cnn"]
    clf = CNN().cuda()
    clf.load_state_dict(torch.load(os.path.join(ROOT, "Q1", "checkpoints", f"q1a_cnn_lr{q1['selected_lr']}.pt")))
    clf.eval()

    dae_d = load_json(os.path.join(HERE, "results", f"q3_ae_d{d_sel}_noise{noise}.json"))["bottleneck"]
    dae = load_ae(f"ae_d{dae_d}_noise{noise}", dae_d)
    b = {"classifier": f"Q1(a) CNN, lr {q1['selected_lr']}", "dae_bottleneck": dae_d, "levels": {}}
    for sigma in [noise / 2, noise, noise * 2]:
        g = torch.Generator(device=xte.device).manual_seed(SEED + 2)
        xn = add_noise(xte, sigma, mean, std, g)
        inputs = {"clean": xte, "noisy": xn, "cleanAE(noisy)": reconstruct(ae_sel, xn),
                  "DAE(noisy)": reconstruct(dae, xn), "cleanAE(clean)": reconstruct(ae_sel, xte),
                  "DAE(clean)": reconstruct(dae, xte)}
        lvl = {}
        for k, x in inputs.items():
            _, acc = evaluate(clf, x, yte)
            mse_pix = (((x - xte) * std) ** 2).mean().item()
            lvl[k] = {"test_acc": acc, "mse_to_clean_pixel": mse_pix}
        b["levels"][str(sigma)] = lvl
        print(f"(b) sigma {sigma}: " + "  ".join(f"{k} {v['test_acc']:.4f}" for k, v in lvl.items()))
    return b


def part_c(data, noise, d_sel):
    std = data["std"]
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    xte, yte = data["test"]
    dae_d = d_sel
    c = {}
    encs = [(f"AE-{d}", f"ae_d{d}", d) for d in BOTTLENECKS] + [(f"DAE-{dae_d}", f"ae_d{dae_d}_noise{noise}", dae_d)]
    for name, tag, d in encs:
        ae = load_ae(tag, d)
        ftr, fva, fte = codes(ae, xtr), codes(ae, xva), codes(ae, xte)
        tn, tp = recon_error(ae, xte, xte, std)
        lp = linear_probe(ftr, ytr, fva, yva, fte, yte)
        kn = knn(ftr, ytr, fva, yva, fte, yte)
        c[name] = {"dim": d, "test_mse": tn, "test_mse_pixel": tp, "test_psnr": float(10 * np.log10(1 / tp)),
                   "linear_probe": lp, "knn": kn}
        print(f"(c) {name}: test mse(pix) {tp:.5f}  linear probe test {lp['test_acc']:.4f}  knn test {kn['test_acc']:.4f}")
    f = lambda x: x.reshape(len(x), -1)
    c["raw pixels"] = {"dim": 3072, "linear_probe": linear_probe(f(xtr), ytr, f(xva), yva, f(xte), yte),
                       "knn": knn(f(xtr), ytr, f(xva), yva, f(xte), yte)}
    print("(c) raw pixels:", c["raw pixels"])
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--parts", default="abc")
    args = ap.parse_args()
    main(args.noise, args.parts)

