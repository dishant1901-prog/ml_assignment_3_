import argparse
import os
import pickle
import sys
import time
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
SEED = 242003
CS = [0.001, 0.01, 0.1, 1, 10, 100]
KERNELS = {
    "linear": dict(kernel="linear"),
    "rbf": dict(kernel="rbf", gamma="scale"),
    "poly3": dict(kernel="poly", degree=3, gamma="scale", coef0=1.0),
}


def extract_features(encoder):
    import torch
    sys.path.insert(0, ROOT)
    from common.data import get_data
    from Q3.models import ConvAE
    from Q3.q3_eval import codes
    d = int(encoder.split("_d")[1].split("_")[0])
    ae = ConvAE(d).cuda()
    ae.load_state_dict(torch.load(os.path.join(ROOT, "Q3", "checkpoints", f"q3_{encoder}.pt")))
    ae.eval()
    data = get_data()
    out = {"encoder": encoder}
    for k in ("train", "val", "test"):
        x, y = data[k]
        out[f"f_{k}"] = codes(ae, x).cpu().numpy()
        out[f"y_{k}"] = y.cpu().numpy()
    mu, sd = out["f_train"].mean(0), out["f_train"].std(0) + 1e-6
    for k in ("train", "val", "test"):
        out[f"f_{k}"] = ((out[f"f_{k}"] - mu) / sd).astype(np.float32)
    os.makedirs(RES, exist_ok=True)
    np.savez(os.path.join(RES, "features.npz"), **out)

    idx = np.arange(len(out["y_train"]))
    sub, _ = train_test_split(idx, train_size=10000, stratify=out["y_train"], random_state=SEED)
    sub.sort()
    np.save(os.path.join(RES, "svm_subset_idx.npy"), sub)
    print("features", {k: v.shape for k, v in out.items() if k.startswith("f_")},
          "subset", len(sub), np.bincount(out["y_train"][sub]))


def load():
    f = np.load(os.path.join(RES, "features.npz"))
    sub = np.load(os.path.join(RES, "svm_subset_idx.npy"))
    return f["f_train"][sub], f["y_train"][sub], f["f_val"], f["y_val"], f["f_test"], f["y_test"]


def fit_one(kname, C, xtr, ytr, xva, yva):
    t = time.time()
    clf = SVC(C=C, cache_size=200, decision_function_shape="ovo", **KERNELS[kname])
    clf.fit(xtr, ytr)
    return {"kernel": kname, "C": C, "train_acc": float((clf.predict(xtr) == ytr).mean()),
            "val_acc": float((clf.predict(xva) == yva).mean()),
            "sv_fraction": float(clf.n_support_.sum() / len(ytr)),
            "n_sv": int(clf.n_support_.sum()), "fit_seconds": time.time() - t}


def sweep(n_jobs=2):
    import json
    xtr, ytr, xva, yva, _, _ = load()
    out_dir = os.path.join(RES, "sweep")
    os.makedirs(out_dir, exist_ok=True)
    todo = [(k, C) for k in KERNELS for C in CS
            if not os.path.exists(os.path.join(out_dir, f"{k}_C{C}.json"))]

    def fit_and_save(k, C):
        r = fit_one(k, C, xtr, ytr, xva, yva)
        json.dump(r, open(os.path.join(out_dir, f"{k}_C{C}.json"), "w"), indent=2)
        print(r, flush=True)

    Parallel(n_jobs=n_jobs, verbose=5)(delayed(fit_and_save)(k, C) for k, C in todo)
    rows = [json.load(open(os.path.join(out_dir, f"{k}_C{C}.json"))) for k in KERNELS for C in CS]
    with open(os.path.join(RES, "svm_sweep.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"{'kernel':7s} {'C':>7s} {'train':>6s} {'val':>6s} {'SV frac':>7s} {'time':>6s}")
    for r in rows:
        print(f"{r['kernel']:7s} {r['C']:7g} {r['train_acc']:.4f} {r['val_acc']:.4f} {r['sv_fraction']:.4f} "
              f"{r['fit_seconds']:6.0f}")


def test():
    import json
    rows = json.load(open(os.path.join(RES, "svm_sweep.json")))
    best = max(rows, key=lambda r: r["val_acc"])
    xtr, ytr, xva, yva, xte, yte = load()
    clf = SVC(C=best["C"], cache_size=200, decision_function_shape="ovo", **KERNELS[best["kernel"]])
    clf.fit(xtr, ytr)
    pred = clf.predict(xte)
    acc = float((pred == yte).mean())
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (yte, pred), 1)
    with open(os.path.join(RES, "svm_selected.pkl"), "wb") as fh:
        pickle.dump(clf, fh)
    json.dump({"selected": best, "test_acc": acc, "confusion": cm.tolist()},
              open(os.path.join(RES, "svm_test.json"), "w"), indent=2)
    print("selected", best, "test acc", acc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["features", "sweep", "test"])
    ap.add_argument("--encoder", default=None)
    a = ap.parse_args()
    if a.step == "features":
        extract_features(a.encoder)
    elif a.step == "sweep":
        sweep()
    else:
        test()

