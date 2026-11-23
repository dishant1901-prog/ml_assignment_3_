import io
import os
import pickle
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "cifar10.npz")


def from_official():
    d = os.path.join(ROOT, "data", "cifar-10-batches-py")
    if not os.path.isdir(d):
        return None

    def load(name):
        with open(os.path.join(d, name), "rb") as f:
            b = pickle.load(f, encoding="bytes")
        return b[b"data"].reshape(-1, 3, 32, 32), np.array(b[b"labels"], dtype=np.int64)

    parts = [load(f"data_batch_{i}") for i in range(1, 6)]
    xtr = np.concatenate([p[0] for p in parts])
    ytr = np.concatenate([p[1] for p in parts])
    xte, yte = load("test_batch")
    return xtr, ytr, xte, yte


def from_parquet():
    import pyarrow.parquet as pq
    from PIL import Image
    d = os.path.join(ROOT, "data", "hf")

    def load(name):
        t = pq.read_table(os.path.join(d, name)).to_pydict()
        x = np.stack([np.array(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in t["img"]])
        return x.transpose(0, 3, 1, 2).copy(), np.array(t["label"], dtype=np.int64)

    xtr, ytr = load("train-00000-of-00001.parquet")
    xte, yte = load("test-00000-of-00001.parquet")
    return xtr, ytr, xte, yte


if __name__ == "__main__":
    if "--compare" in sys.argv:
        a, b = from_official(), from_parquet()
        for name, u, v in zip(["x_train", "y_train", "x_test", "y_test"], a, b):
            print(name, u.shape, "identical:", np.array_equal(u, v))
        sys.exit()

    if "--official" in sys.argv:
        data, src = from_official(), "official batches"
    else:
        data, src = from_parquet(), "huggingface parquet"
    xtr, ytr, xte, yte = data
    assert xtr.shape == (50000, 3, 32, 32) and xte.shape == (10000, 3, 32, 32)
    np.savez(OUT, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)
    print("source:", src)
    print("train", xtr.shape, np.bincount(ytr), "test", xte.shape, np.bincount(yte))

