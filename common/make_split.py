import numpy as np
from common.data import make_split, load_cifar, SPLIT_FILE, SPLIT_SEED

if __name__ == "__main__":
    tr, va = make_split()
    _, y, _, _ = load_cifar()
    print("seed", SPLIT_SEED, "->", SPLIT_FILE)
    print("train", len(tr), "val", len(va), "overlap", len(np.intersect1d(tr, va)))
    print("train per class", np.bincount(y[tr]))
    print("val per class  ", np.bincount(y[va]))

