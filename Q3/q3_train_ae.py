import argparse
import os
import time
import torch
import torch.nn.functional as F

from common.data import get_data, batches
from common.utils import set_seed, count_params, save_json
from Q3.models import ConvAE

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 242003
CONFIG = {"epochs": 30, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-5, "optimizer": "AdamW",
          "schedule": "OneCycle, 15% warmup", "loss": "MSE (normalised space)"}


def add_noise(x, sigma, mean, std, gen=None):
    pix = x * std + mean
    noise = torch.randn(pix.shape, device=x.device, generator=gen) * sigma
    return ((pix + noise).clamp(0, 1) - mean) / std


@torch.no_grad()
def recon_error(ae, x_in, x_target, std, bs=1000):
    ae.eval()
    se_n, se_p = 0.0, 0.0
    for i in range(0, len(x_in), bs):
        r = ae(x_in[i:i + bs])
        d = r - x_target[i:i + bs]
        se_n += (d ** 2).sum().item()
        se_p += ((d * std) ** 2).sum().item()
    n = x_in.numel()
    return se_n / n, se_p / n


def train_ae(d, noise, data):
    xtr, _ = data["train"]
    xva, _ = data["val"]
    mean, std = data["mean"], data["std"]
    set_seed(SEED)
    ae = ConvAE(d).cuda()
    opt = torch.optim.AdamW(ae.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    steps = (len(xtr) + CONFIG["batch_size"] - 1) // CONFIG["batch_size"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG["lr"], epochs=CONFIG["epochs"],
                                                steps_per_epoch=steps, pct_start=0.15)
    gen = torch.Generator(device=xtr.device).manual_seed(SEED)
    xva_in = add_noise(xva, noise, mean, std, torch.Generator(device=xtr.device).manual_seed(SEED + 1)) \
        if noise > 0 else xva

    hist = {"epoch": [], "train_mse": [], "val_mse": [], "val_mse_pixel": []}
    t0 = time.time()
    for ep in range(1, CONFIG["epochs"] + 1):
        ae.train()
        tot, n = 0.0, 0
        for xb, _ in batches(xtr, xtr[:, 0, 0, 0], CONFIG["batch_size"], gen=gen):
            inp = add_noise(xb, noise, mean, std, gen) if noise > 0 else xb
            loss = F.mse_loss(ae(inp), xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        vn, vp = recon_error(ae, xva_in, xva, std)
        hist["epoch"].append(ep)
        hist["train_mse"].append(tot / n)
        hist["val_mse"].append(vn)
        hist["val_mse_pixel"].append(vp)
        print(f"d={d} noise={noise} ep {ep:2d}  train {tot / n:.4f}  val {vn:.4f}  val(pixel) {vp:.5f}  "
              f"{time.time() - t0:.0f}s", flush=True)

    tag = f"ae_d{d}" + (f"_noise{noise}" if noise > 0 else "")
    save_json({"bottleneck": d, "noise": noise, "params": count_params(ae), "config": CONFIG, "history": hist},
              os.path.join(HERE, "results", f"q3_{tag}.json"))
    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    torch.save(ae.state_dict(), os.path.join(HERE, "checkpoints", f"q3_{tag}.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bottleneck", type=int, nargs="+", required=True)
    ap.add_argument("--noise", type=float, default=0.0)
    args = ap.parse_args()
    data = get_data()
    data.pop("test")
    for d in args.bottleneck:
        train_ae(d, args.noise, data)

