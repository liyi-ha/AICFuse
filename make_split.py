# -*- coding: utf-8 -*-
"""Split the 180 original FLIR pairs before any online augmentation."""
import csv
import random
import argparse
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect(folder):
    result = {}
    for path in Path(folder).rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            if path.stem in result:
                raise RuntimeError(f"Duplicate stem {path.stem} in {folder}")
            result[path.stem] = str(path.resolve())
    return result


def write_csv(path, keys, ir_map, vis_map):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "ir", "vis"])
        writer.writeheader()
        writer.writerows(
            {"key": key, "ir": ir_map[key], "vis": vis_map[key]}
            for key in keys
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--split_root", default="./splits_flir180_clean")
    p.add_argument("--split_name", default="flir180_clean")
    p.add_argument("--split_seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_root)
    ir_map = collect(data_root / "IR")
    vis_map = collect(data_root / "VIS")
    if set(ir_map) != set(vis_map):
        raise RuntimeError("IR/VIS stems are not paired")

    original_keys = [f"{i:04d}" for i in range(1, 181)]
    missing = [
        key for key in original_keys
        if key not in ir_map or key not in vis_map
    ]
    if missing:
        raise RuntimeError(f"Missing original FLIR pairs: {missing[:10]}")
    if len(ir_map) != 540:
        raise RuntimeError(f"Expected 540 stored pairs, found {len(ir_map)}")

    keys = original_keys.copy()
    random.Random(args.split_seed).shuffle(keys)
    train_keys = keys[:162]
    val_keys = keys[162:]

    split_root = Path(args.split_root)
    split_root.mkdir(parents=True, exist_ok=True)
    train_csv = split_root / (
        f"{args.split_name}_splitseed{args.split_seed}_train.csv"
    )
    val_csv = split_root / (
        f"{args.split_name}_splitseed{args.split_seed}_val.csv"
    )

    if train_csv.exists() and val_csv.exists() and not args.force:
        print("[KEEP] clean original-scene split already exists")
    else:
        write_csv(train_csv, train_keys, ir_map, vis_map)
        write_csv(val_csv, val_keys, ir_map, vis_map)

    if set(train_keys) & set(val_keys):
        raise RuntimeError("Original-scene leakage detected")

    print("=" * 72)
    print("FLIR180 CLEAN ORIGINAL-SCENE SPLIT")
    print("=" * 72)
    print(f"Data root       : {data_root}")
    print(f"Stored pairs    : {len(ir_map)} (180 original + 360 offline aug)")
    print("Used originals  : 180")
    print(f"Split seed      : {args.split_seed}")
    print(f"Train originals : {len(train_keys)}")
    print(f"Val originals   : {len(val_keys)}")
    print("Offline aug used: 0")
    print(f"Train CSV       : {train_csv}")
    print(f"Val CSV         : {val_csv}")
    print("[PASS] split-before-augmentation; no scene leakage.")


if __name__ == "__main__":
    main()
