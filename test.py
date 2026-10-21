# -*- coding: utf-8 -*-
"""
Evaluate AICFuse on TNO, NIR and FLIR.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import pure_models as models

from external_datasets import DATASETS, get_pairs

FUSION_MODES = ("Sum", "Average", "l1_norm")


def read_gray(path, device):
    arr = np.asarray(
        Image.open(path).convert("L"),
        dtype=np.float32,
    ) / 255.0
    return torch.from_numpy(arr)[None, None].to(device)


def l1_addition(y1, y2, window_width=1):
    kernel_size = 2 * window_width + 1
    activity1 = F.avg_pool2d(
        y1.abs().sum(dim=1, keepdim=True),
        kernel_size,
        stride=1,
        padding=window_width,
    )
    activity2 = F.avg_pool2d(
        y2.abs().sum(dim=1, keepdim=True),
        kernel_size,
        stride=1,
        padding=window_width,
    )
    total = activity1 + activity2 + 1e-8
    return activity1 / total * y1 + activity2 / total * y2


def fuse_features(ir_features, vis_features, mode):
    if mode == "Sum":
        return tuple(i + v for i, v in zip(ir_features, vis_features))
    if mode == "Average":
        return tuple((i + v) / 2.0 for i, v in zip(ir_features, vis_features))
    if mode == "l1_norm":
        return tuple(
            l1_addition(i, v)
            for i, v in zip(ir_features, vis_features)
        )
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=models.VARIANTS)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="./results")
    p.add_argument(
        "--checkpoint", choices=["best", "last"], default="best"
    )
    p.add_argument(
        "--dataset",
        choices=list(DATASETS) + ["all"],
        default="all",
    )
    p.add_argument("--mode", choices=FUSION_MODES, default="Sum")
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    enc, dec = models.build(args.variant)
    enc = enc.to(device)
    dec = dec.to(device)

    run_dir = Path(args.out) / args.variant / f"seed_{args.seed}"
    enc_ckpt = torch.load(
        run_dir / f"Encoder_{args.checkpoint}.pkl",
        map_location=device,
        weights_only=False,
    )
    dec_ckpt = torch.load(
        run_dir / f"Decoder_{args.checkpoint}.pkl",
        map_location=device,
        weights_only=False,
    )
    enc.load_state_dict(enc_ckpt["weight"], strict=True)
    dec.load_state_dict(dec_ckpt["weight"], strict=True)
    enc.eval()
    dec.eval()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)

    for dataset in datasets:
        pairs = get_pairs(dataset)
        out_dir = run_dir / f"{dataset}_{args.mode}_{args.checkpoint}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[TEST] {args.variant} seed={args.seed} "
            f"{dataset} {args.mode} {args.checkpoint} pairs={len(pairs)}"
        )

        for pos, (idx, ir_path, vis_path) in enumerate(pairs, start=1):
            ir = read_gray(ir_path, device)
            vis = read_gray(vis_path, device)

            if ir.shape != vis.shape:
                raise RuntimeError(
                    f"{dataset} pair {idx}: "
                    f"IR={tuple(ir.shape)} VIS={tuple(vis.shape)}"
                )

            with torch.no_grad():
                fi = enc(ir)
                fv = enc(vis)
                ff = fuse_features(fi, fv, args.mode)
                fused = dec(*ff).clamp(0, 1)

            img = np.rint(
                fused[0, 0].cpu().numpy() * 255.0
            ).astype(np.uint8)
            Image.fromarray(img).save(out_dir / f"F{idx:04d}.png")

            if pos == 1 or pos % 10 == 0 or pos == len(pairs):
                print(f"  {pos:03d}/{len(pairs):03d}")

        print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
