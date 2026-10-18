# -*- coding: utf-8 -*-
"""
external_datasets.py
Three external test sets under ./test_datasets.

Expected sequential pairs:
  TNO  = 40
  NIR  = 52
  FLIR = 40

Each dataset must contain ir/ and vi/ folders. Images are paired by the
same numeric filename stem, for example ir/1.jpg and vi/1.jpg.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "test_datasets"
DATASETS = ("TNO", "NIR", "FLIR")
EXPECTED = {"TNO": 40, "NIR": 52, "FLIR": 40}
EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def collect_numeric(folder):
    mapping = {}
    for path in Path(folder).iterdir():
        if not path.is_file() or path.suffix.lower() not in EXTS:
            continue
        if not path.stem.isdigit():
            raise RuntimeError(f"Non-numeric image stem in {folder}: {path.name}")
        idx = int(path.stem)
        if idx in mapping:
            raise RuntimeError(f"Duplicate image index {idx} in {folder}")
        mapping[idx] = path
    return mapping


def get_pairs(dataset):
    if dataset not in DATASETS:
        raise ValueError(dataset)

    folder = ROOT / dataset
    ir_dir = folder / "ir"
    vis_dir = folder / "vi"
    if not ir_dir.is_dir() or not vis_dir.is_dir():
        raise FileNotFoundError(f"Expected {ir_dir} and {vis_dir}")

    ir_map = collect_numeric(ir_dir)
    vis_map = collect_numeric(vis_dir)
    if set(ir_map) != set(vis_map):
        missing_vis = sorted(set(ir_map) - set(vis_map))
        missing_ir = sorted(set(vis_map) - set(ir_map))
        raise RuntimeError(
            f"{dataset}: unpaired data; missing vi={missing_vis[:10]}, "
            f"missing ir={missing_ir[:10]}"
        )

    indices = sorted(ir_map)
    pairs = [(idx, ir_map[idx], vis_map[idx]) for idx in indices]

    expected = EXPECTED[dataset]
    if len(pairs) != expected:
        raise RuntimeError(
            f"{dataset}: expected {expected} pairs, found {len(pairs)} "
            f"in {folder}"
        )

    expected_indices = list(range(1, expected + 1))
    if indices != expected_indices:
        raise RuntimeError(
            f"{dataset}: expected consecutive indices 1..{expected}, "
            f"found {indices[:5]}...{indices[-5:]}"
        )

    return pairs


if __name__ == "__main__":
    for ds in DATASETS:
        pairs = get_pairs(ds)
        print(
            f"{ds:4s}: {len(pairs)} pairs | "
            f"first=({pairs[0][1].name},{pairs[0][2].name}) | "
            f"last=({pairs[-1][1].name},{pairs[-1][2].name})"
        )
