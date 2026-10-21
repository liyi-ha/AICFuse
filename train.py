# -*- coding: utf-8 -*-
"""
AICFuse training and validation.

Split seed is fixed (default 2026).
Training seed controls initialization/augmentation only.

Best checkpoint is selected using a fixed validation objective:
  0.6*recon + 1.0*base + 1.0*detail + 0.3*texture + 0.5*align
"""
import csv
import json
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF

import pure_models as models

from losses import (
    TextureGradientLoss,
    DetailAlignmentLoss,
    DynamicLossScheduler,
    normalize_features,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    s = torch.initial_seed() % (2 ** 32)
    random.seed(s)
    np.random.seed(s)


def read_split(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty split: {csv_path}")
    return rows


class PairDataset(Dataset):
    def __init__(self, split_csv, crop=128, training=True, repeat=1):
        self.rows = read_split(split_csv)
        self.crop = crop
        self.training = training
        self.repeat = repeat
        if self.repeat < 1:
            raise ValueError("repeat must be >= 1")

    def __len__(self):
        return len(self.rows) * self.repeat

    def __getitem__(self, i):
        row = self.rows[i % len(self.rows)]
        ir = Image.open(row["ir"]).convert("L")
        vis = Image.open(row["vis"]).convert("L")

        if ir.size != vis.size:
            raise RuntimeError(
                f"{row['key']} size mismatch: IR={ir.size}, VIS={vis.size}"
            )
        if ir.width < self.crop or ir.height < self.crop:
            raise RuntimeError(
                f"{row['key']} smaller than crop {self.crop}: {ir.size}"
            )

        if self.training:
            top, left, h, w = RandomCrop.get_params(
                vis, output_size=(self.crop, self.crop)
            )
            ir = TF.crop(ir, top, left, h, w)
            vis = TF.crop(vis, top, left, h, w)
            if random.random() < 0.5:
                ir = TF.hflip(ir)
                vis = TF.hflip(vis)
        else:
            ir = TF.center_crop(ir, [self.crop, self.crop])
            vis = TF.center_crop(vis, [self.crop, self.crop])

        return TF.to_tensor(ir), TF.to_tensor(vis)


def components(enc, dec, ir, vis, l1, tex, align):
    vf = enc(vis)
    inf = enc(ir)
    rv = dec(*vf)
    ri = dec(*inf)

    return {
        "recon": l1(vis, rv) + l1(ir, ri),
        "base": l1(
            normalize_features(inf[2]),
            normalize_features(vf[2]),
        ),
        "detail": l1(
            normalize_features(inf[3]),
            normalize_features(vf[3]),
        ),
        "align": align(inf[3], vf[3]),
        "texture": (tex(vis, rv) + tex(ir, ri)) / 2.0,
    }


def total_loss(p, w):
    return (
        w["recon"] * p["recon"]
        + w["base"] * p["base"]
        + w["detail"] * p["detail"]
        + w["align"] * p["align"]
        + w["texture"] * p["texture"]
    )


def val_epoch(enc, dec, loader, device, l1, tex, align):
    enc.eval()
    dec.eval()

    w = {
        "recon": 0.6,
        "base": 1.0,
        "detail": 1.0,
        "texture": 0.3,
        "align": 0.5,
    }
    sums = {k: 0.0 for k in (
        "val_total", "val_recon", "val_base",
        "val_detail", "val_align", "val_texture"
    )}
    n = 0

    with torch.no_grad():
        for ir, vis in loader:
            ir = ir.to(device, non_blocking=True)
            vis = vis.to(device, non_blocking=True)
            p = components(enc, dec, ir, vis, l1, tex, align)
            t = total_loss(p, w)
            batch_size = ir.shape[0]

            sums["val_total"] += t.item() * batch_size
            sums["val_recon"] += p["recon"].item() * batch_size
            sums["val_base"] += p["base"].item() * batch_size
            sums["val_detail"] += p["detail"].item() * batch_size
            sums["val_align"] += p["align"].item() * batch_size
            sums["val_texture"] += p["texture"].item() * batch_size
            n += batch_size

    return {k: v / n for k, v in sums.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=models.VARIANTS)
    p.add_argument("--split_root", default="./splits_flir180_clean")
    p.add_argument("--split_name", default="flir180_clean")
    p.add_argument("--split_seed", type=int, default=2026)
    p.add_argument("--expected_train_pairs", type=int, default=486)
    p.add_argument("--expected_val_pairs", type=int, default=18)
    p.add_argument("--train_repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--out", default="./results")
    args = p.parse_args()

    set_seed(args.seed)

    train_csv = (
        Path(args.split_root)
        / f"{args.split_name}_splitseed{args.split_seed}_train.csv"
    )
    val_csv = (
        Path(args.split_root)
        / f"{args.split_name}_splitseed{args.split_seed}_val.csv"
    )

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            "Fixed split missing. Run python make_split.py first."
        )

    train_ds = PairDataset(
        train_csv,
        crop=args.crop,
        training=True,
        repeat=args.train_repeat,
    )
    val_ds = PairDataset(val_csv, crop=args.crop, training=False)
    if (
        len(train_ds) != args.expected_train_pairs
        or len(val_ds) != args.expected_val_pairs
    ):
        raise RuntimeError(
            f"Expected fixed split "
            f"{args.expected_train_pairs}/{args.expected_val_pairs}, found "
            f"{len(train_ds)}/{len(val_ds)}"
        )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    enc, dec = models.build(args.variant)
    enc = enc.to(device)
    dec = dec.to(device)

    params = models.nparams(enc) + models.nparams(dec)

    print("=" * 78)
    print(f"Variant        : {args.variant}")
    print(f"Train seed     : {args.seed}")
    print(f"Split seed     : {args.split_seed}")
    print(f"Device         : {device}")
    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Train pairs    : {len(train_ds)}")
    print(f"Train originals: {len(train_ds.rows)}")
    print(f"Train repeat   : {args.train_repeat}")
    print(f"Val pairs      : {len(val_ds)}")
    print(f"Train batches  : {len(train_loader)}")
    print(f"Val batches    : {len(val_loader)}")
    print(f"Params         : {params:,}")
    print("=" * 78)

    l1 = nn.L1Loss()
    tex = TextureGradientLoss(
        texture_weight=5.0, gradient_weight=10.0
    ).to(device)
    align = DetailAlignmentLoss().to(device)
    dyn = DynamicLossScheduler(args.epochs)

    oe = optim.Adam(enc.parameters(), lr=args.lr)
    od = optim.Adam(dec.parameters(), lr=args.lr)
    se = optim.lr_scheduler.CosineAnnealingLR(
        oe, T_max=args.epochs, eta_min=1e-6
    )
    sd = optim.lr_scheduler.CosineAnnealingLR(
        od, T_max=args.epochs, eta_min=1e-6
    )

    run_dir = (
        Path(args.out)
        / args.variant
        / f"seed_{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "pure_interpolation",
            "variant": args.variant,
            "train_seed": args.seed,
            "split_root": str(Path(args.split_root).resolve()),
            "split_name": args.split_name,
            "split_seed": args.split_seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "crop": args.crop,
            "lr": args.lr,
            "grad_clip": args.grad_clip,
            "train_pairs": len(train_ds),
            "train_original_pairs": len(train_ds.rows),
            "train_repeat": args.train_repeat,
            "val_pairs": len(val_ds),
            "params": params,
        }, f, indent=2, ensure_ascii=False)

    best_val = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        enc.train()
        dec.train()

        wr = dyn.get_weights(epoch)
        tw = {
            "recon": wr["recon_weight"],
            "base": wr["base_weight"],
            "detail": wr["detail_weight"],
            "texture": wr["texture_weight"],
            "align": wr["align_weight"],
        }

        sums = {k: 0.0 for k in (
            "train_total", "train_recon", "train_base",
            "train_detail", "train_align", "train_texture"
        )}
        n = 0

        for ir, vis in train_loader:
            ir = ir.to(device, non_blocking=True)
            vis = vis.to(device, non_blocking=True)
            batch_size = ir.shape[0]

            oe.zero_grad(set_to_none=True)
            od.zero_grad(set_to_none=True)

            p = components(enc, dec, ir, vis, l1, tex, align)
            t = total_loss(p, tw)
            t.backward()

            nn.utils.clip_grad_norm_(enc.parameters(), args.grad_clip)
            nn.utils.clip_grad_norm_(dec.parameters(), args.grad_clip)
            oe.step()
            od.step()

            sums["train_total"] += t.item() * batch_size
            sums["train_recon"] += p["recon"].item() * batch_size
            sums["train_base"] += p["base"].item() * batch_size
            sums["train_detail"] += p["detail"].item() * batch_size
            sums["train_align"] += p["align"].item() * batch_size
            sums["train_texture"] += p["texture"].item() * batch_size
            n += batch_size

        train_avg = {k: v / n for k, v in sums.items()}
        val_avg = val_epoch(
            enc, dec, val_loader, device, l1, tex, align
        )

        if val_avg["val_total"] < best_val:
            best_val = val_avg["val_total"]
            best_epoch = epoch
            torch.save({
                "weight": enc.state_dict(),
                "variant": args.variant,
                "seed": args.seed,
                "epoch": epoch,
                "val_total": best_val,
            }, run_dir / "Encoder_best.pkl")
            torch.save({
                "weight": dec.state_dict(),
                "variant": args.variant,
                "seed": args.seed,
                "epoch": epoch,
                "val_total": best_val,
            }, run_dir / "Decoder_best.pkl")

        se.step()
        sd.step()

        row = {
            "epoch": epoch,
            "lr": oe.param_groups[0]["lr"],
            "train_detail_weight": tw["detail"],
            **train_avg,
            **val_avg,
            "best_val_so_far": best_val,
            "best_epoch_so_far": best_epoch,
        }
        history.append(row)

        print(
            f"E{epoch:03d}/{args.epochs} "
            f"train={row['train_total']:.6f} "
            f"val={row['val_total']:.6f} "
            f"best={best_val:.6f}@E{best_epoch:03d} "
            f"lr={row['lr']:.2e}"
        )

    torch.save(
        {"weight": enc.state_dict(), "epoch": args.epochs},
        run_dir / "Encoder_last.pkl"
    )
    torch.save(
        {"weight": dec.state_dict(), "epoch": args.epochs},
        run_dir / "Decoder_last.pkl"
    )

    with open(
        run_dir / "history.csv", "w",
        newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    with open(run_dir / "best.txt", "w", encoding="utf-8") as f:
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_total={best_val:.10f}\n")

    print(f"[BEST] E{best_epoch} val_total={best_val:.6f}")
    print(f"[DONE] {run_dir}")


if __name__ == "__main__":
    main()
