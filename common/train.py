import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.data import batches, augment


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    model.eval()
    loss_sum, correct = 0.0, 0
    for xb, yb in batches(x, y, batch_size, shuffle=False):
        out = model(xb)
        loss_sum += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return loss_sum / len(y), correct / len(y)


@torch.no_grad()
def predict(model, x, batch_size=1000):
    model.eval()
    probs = []
    for i in range(0, x.shape[0], batch_size):
        probs.append(F.softmax(model(x[i:i + batch_size]), dim=1))
    return torch.cat(probs)


def train(model, data, epochs, lr, weight_decay=5e-4, batch_size=128, use_aug=False,
          opt_name="sgd", schedule="onecycle", seed=0, log_every=1, evals_per_epoch=None,
          extra_eval=None, verbose=True):
    torch.backends.cudnn.benchmark = True
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    gen = torch.Generator(device=xtr.device)
    gen.manual_seed(seed)

    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=weight_decay, nesterov=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = (len(ytr) + batch_size - 1) // batch_size
    if schedule == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                    steps_per_epoch=steps_per_epoch, pct_start=0.15)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    hist = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "time": []}
    t0 = time.time()

    def log(ep):
        tl, ta = evaluate(model, xtr, ytr)
        vl, va = evaluate(model, xva, yva)
        hist["epoch"].append(ep)
        hist["train_loss"].append(tl)
        hist["train_acc"].append(ta)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        hist["time"].append(time.time() - t0)
        if extra_eval is not None:
            for k, v in extra_eval(model, ep).items():
                hist.setdefault(k, []).append(v)
        if verbose:
            print(f"ep {ep:7.3f}  train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        model.train()

    eval_steps = set()
    if evals_per_epoch:
        total = epochs * steps_per_epoch
        k = steps_per_epoch / evals_per_epoch
        eval_steps = {round(i * k) for i in range(1, int(total / k) + 1)}
        log(0.0)

    step = 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in batches(xtr, ytr, batch_size, shuffle=True, gen=gen):
            if use_aug:
                xb = augment(xb)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step in eval_steps:
                log(round(step / steps_per_epoch, 3))

        if not evals_per_epoch and (ep % log_every == 0 or ep == epochs):
            log(ep)
    return hist

