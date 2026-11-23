import argparse
import itertools
import json
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from Q4.q4_svm import RES, ROOT, SEED, load

FIG = os.path.join(os.path.dirname(RES), "figures")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def raw_images():
    d = np.load(os.path.join(ROOT, "data", "cifar10.npz"))
    s = np.load(os.path.join(ROOT, "common", "split_seed242003.npz"))
    sub = np.load(os.path.join(RES, "svm_subset_idx.npy"))
    train_imgs = d["x_train"][s["train_idx"]][sub]
    return train_imgs.transpose(0, 2, 3, 1), d["x_test"].transpose(0, 2, 3, 1)


def pair_column(i, j):
    return list(itertools.combinations(range(10), 2)).index((i, j))


def pair_dual_coef(clf, i, j):
    starts = np.concatenate([[0], np.cumsum(clf.n_support_)])
    sv_i = np.arange(starts[i], starts[i + 1])
    sv_j = np.arange(starts[j], starts[j + 1])
    return sv_i, clf.dual_coef_[j - 1, sv_i], sv_j, clf.dual_coef_[i, sv_j]


def pairs():
    clf = pickle.load(open(os.path.join(RES, "svm_selected.pkl"), "rb"))
    info = json.load(open(os.path.join(RES, "svm_test.json")))
    cm = np.array(info["confusion"])
    xtr, ytr, _, _, xte, yte = load()
    imgs_tr, imgs_te = raw_images()
    dec_te = clf.decision_function(xte)
    pred_te = clf.predict(xte)

    sym = cm + cm.T
    np.fill_diagonal(sym, 0)
    order = sorted(((sym[i, j], i, j) for i in range(10) for j in range(i + 1, 10)), reverse=True)

    mu = np.stack([xtr[ytr == c].mean(0) for c in range(10)])
    spread = np.array([((xtr[ytr == c] - mu[c]) ** 2).sum(1).mean() for c in range(10)])
    geo = {}
    for _, i, j in order:
        sep = ((mu[i] - mu[j]) ** 2).sum() / ((spread[i] + spread[j]) / 2)
        sv_i, a_i, sv_j, a_j = pair_dual_coef(clf, i, j)
        n_sv_pair = int((a_i != 0).sum() + (a_j != 0).sum())
        n_pair = int(((ytr == i) | (ytr == j)).sum())
        m = (yte == i) | (yte == j)
        col = pair_column(i, j)
        pair_pred = np.where(dec_te[m, col] > 0, i, j)
        geo[f"{i}-{j}"] = {"classes": [CLASSES[i], CLASSES[j]], "test_confusions": int(sym[i, j]),
                           "separation_ratio": float(sep), "pair_sv_fraction": n_sv_pair / n_pair,
                           "pairwise_test_acc": float((pair_pred == yte[m]).mean())}
    seps = np.array([g["separation_ratio"] for g in geo.values()])
    confs = np.array([g["test_confusions"] for g in geo.values()])
    rho = float(np.corrcoef(np.argsort(np.argsort(seps)), np.argsort(np.argsort(confs)))[0, 1])

    os.makedirs(FIG, exist_ok=True)
    hard = order[:3]
    report = {"spearman_separation_vs_confusions": rho, "all_pairs": geo, "hard_pairs": []}
    for _, i, j in hard:
        col = pair_column(i, j)
        sv_i, a_i, sv_j, a_j = pair_dual_coef(clf, i, j)
        tr_idx = clf.support_
        cand = []
        for svs, coefs in ((sv_i, a_i), (sv_j, a_j)):
            ids = tr_idx[svs[coefs != 0]]
            s = clf.decision_function(xtr[ids])[:, col]
            k = np.argsort(np.abs(s))[:4]
            cand.append(list(zip(ids[k], s[k])))
        m = np.where((yte == i) | (yte == j))[0]
        s = dec_te[m, col]
        near = m[np.argsort(np.abs(s))[:8]]

        fig, axes = plt.subplots(2, 8, figsize=(6.9, 2.1))
        for ax, (idx, sc) in zip(axes[0], cand[0] + cand[1]):
            ax.imshow(imgs_tr[idx])
            ax.set_title(f"{CLASSES[ytr[idx]][:5]} {sc:+.3f}", fontsize=6)
        for ax, idx in zip(axes[1], near):
            ax.imshow(imgs_te[idx])
            ok = "" if pred_te[idx] == yte[idx] else " x"
            ax.set_title(f"{CLASSES[yte[idx]][:5]} {dec_te[idx, col]:+.3f}{ok}", fontsize=6,
                         color="#d03b3b" if ok else "#0b0b0b")
        for ax in axes.ravel():
            ax.axis("off")
        axes[0, 0].text(-40, 16, "train\nSVs", fontsize=6, va="center")
        axes[1, 0].text(-40, 16, "test,\nnear", fontsize=6, va="center")
        fig.suptitle(f"{CLASSES[i]} (+) vs {CLASSES[j]} (-): pair decision scores", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"q4b_pair_{CLASSES[i]}_{CLASSES[j]}.png"), dpi=220)
        plt.close(fig)
        report["hard_pairs"].append({"pair": [CLASSES[i], CLASSES[j]], **geo[f"{i}-{j}"],
                                     "near_test_errors": int((pred_te[near] != yte[near]).sum()),
                                     "near_test_scores": [float(dec_te[t, col]) for t in near],
                                     "sv_scores": [float(x[1]) for x in cand[0] + cand[1]]})
        print(CLASSES[i], CLASSES[j], geo[f"{i}-{j}"])
    print("spearman(separation ratio, confusions) =", rho)
    json.dump(report, open(os.path.join(RES, "svm_pairs.json"), "w"), indent=2)


CANDIDATES = [("kmeans", k) for k in (5, 10, 15, 20, 30)] + [("gmm", k) for k in (5, 10, 15, 20, 30)]


def make_model(method, k):
    if method == "kmeans":
        return KMeans(k, n_init=10, random_state=SEED)
    return GaussianMixture(k, covariance_type="diag", n_init=3, random_state=SEED, max_iter=300)


def full_train():
    f = np.load(os.path.join(RES, "features.npz"))
    return f["f_train"], f["y_train"], f["f_val"], f["y_val"], f["f_test"], f["y_test"]


def cluster_select():
    xtr, _, xva, yva, _, _ = full_train()
    rows = []
    rng = np.random.RandomState(SEED)
    sil_idx = rng.choice(len(xva), 3000, replace=False)
    for method, k in CANDIDATES:
        m = make_model(method, k).fit(xtr)
        pva = m.predict(xva)
        rows.append({"method": method, "k": k, "val_ari": float(adjusted_rand_score(yva, pva)),
                     "val_nmi": float(normalized_mutual_info_score(yva, pva)),
                     "val_silhouette": float(silhouette_score(xva[sil_idx], pva[sil_idx]))})
        print(rows[-1], flush=True)
    json.dump(rows, open(os.path.join(RES, "cluster_select.json"), "w"), indent=2)


def cluster_test():
    rows = json.load(open(os.path.join(RES, "cluster_select.json")))
    best = max(rows, key=lambda r: r["val_ari"])
    xtr, _, _, _, xte, yte = full_train()
    m = make_model(best["method"], best["k"]).fit(xtr)
    pte = m.predict(xte)
    cont = np.zeros((10, best["k"]), dtype=int)
    np.add.at(cont, (yte, pte), 1)
    out = {"selected": best, "test_ari": float(adjusted_rand_score(yte, pte)),
           "test_nmi": float(normalized_mutual_info_score(yte, pte)), "contingency": cont.tolist(),
           "test_assignment": pte.tolist()}
    json.dump(out, open(os.path.join(RES, "cluster_test.json"), "w"), indent=2)
    print("selected", best, "test ARI", out["test_ari"], "test NMI", out["test_nmi"])
    print(cont)


def compare():
    _, _, _, _, xte, yte = load()
    ct = json.load(open(os.path.join(RES, "cluster_test.json")))
    pte = np.array(ct["test_assignment"])
    pairs_info = json.load(open(os.path.join(RES, "svm_pairs.json")))["all_pairs"]
    cont = np.array(ct["contingency"])
    rows = []
    for key, g in pairs_info.items():
        i, j = map(int, key.split("-"))
        sep = np.maximum(cont[i], cont[j]).sum() / (cont[i].sum() + cont[j].sum())
        mixed = [c for c in range(cont.shape[1]) if min(cont[i, c], cont[j, c]) >= 0.25 * (cont[i, c] + cont[j, c])
                 and cont[i, c] + cont[j, c] > 0]
        frac_mixed = (cont[i, mixed].sum() + cont[j, mixed].sum()) / (cont[i].sum() + cont[j].sum())
        rows.append({"pair": g["classes"], "svm_pairwise_acc": g["pairwise_test_acc"],
                     "cluster_pairwise_acc": float(sep), "frac_in_mixed_clusters": float(frac_mixed),
                     "difference": float(g["pairwise_test_acc"] - sep)})
    rows.sort(key=lambda r: -abs(r["difference"]))
    per_class = []
    for c in range(10):
        purity_share = cont[c].max() / cont[c].sum()
        per_class.append({"class": CLASSES[c], "largest_cluster_share": float(purity_share),
                          "dominant_cluster": int(cont[c].argmax())})
    json.dump({"pairs": rows, "per_class": per_class}, open(os.path.join(RES, "svm_vs_cluster.json"), "w"),
              indent=2)
    for r in rows[:8]:
        print(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pairs", "cluster-select", "cluster-test", "compare"])
    s = ap.parse_args().step
    {"pairs": pairs, "cluster-select": cluster_select, "cluster-test": cluster_test, "compare": compare}[s]()

