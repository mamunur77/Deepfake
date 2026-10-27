"""
src.data.build_ai_face_splits  ·  Auto-discovers real & fake face datasets on Kaggle/local
and creates balanced train/val/test splits without requiring manual paths.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path

import pandas as pd


def _save_df(df: pd.DataFrame, out_dir: str, split_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{split_name}.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}: {len(df):,} items (Real: {(df['label'] == 0).sum():,}, Fake: {(df['label'] == 1).sum():,})")

    # Mirror to alternate Kaggle paths so relative and absolute lookups never fail
    for alt_parent in ["/kaggle/working/deepfake-vit", "/kaggle/working"]:
        if os.path.exists(alt_parent):
            alt_dir = os.path.join(alt_parent, "data", "splits")
            os.makedirs(alt_dir, exist_ok=True)
            df.to_csv(os.path.join(alt_dir, f"{split_name}.csv"), index=False)


def auto_build_splits(
    input_root: str = "/kaggle/input",
    out_dir: str = "data/splits",
    train_per_class: int = 5000,
    val_per_class: int = 1000,
    test_per_class: int = 1000,
    seed: int = 42,
) -> None:
    random.seed(seed)
    print(f"[splits] Searching for real and fake face images under {input_root}...")

    real_dirs: list[str] = []
    fake_dirs: list[str] = []

    for root, dirs, _ in os.walk(input_root):
        for d in dirs:
            d_l = d.lower()
            if "real" in d_l and "fake" not in d_l:
                real_dirs.append(os.path.join(root, d))
            elif "fake" in d_l:
                fake_dirs.append(os.path.join(root, d))

    print(f"[splits] Found {len(real_dirs)} 'real' folders and {len(fake_dirs)} 'fake' folders.")
    for rd in real_dirs[:3]:
        print(f"  -> {rd}")

    if not real_dirs or not fake_dirs:
        raise FileNotFoundError(
            f"Could not find 'real' and 'fake' folders under {input_root}. "
            "Please check attached datasets."
        )

    os.makedirs(out_dir, exist_ok=True)
    has_splits = any("train" in r.lower() for r in real_dirs)

    if has_splits:
        print("[splits] Standard train/valid/test structure detected!")
        split_configs = [
            ("train", ["train"], train_per_class),
            ("val", ["val", "valid"], val_per_class),
            ("test", ["test"], test_per_class),
        ]
        for split_name, keywords, n_samples in split_configs:
            r_dir = next((d for d in real_dirs if any(k in d.lower() for k in keywords)), None)
            f_dir = next((d for d in fake_dirs if any(k in d.lower() for k in keywords)), None)

            r_files = glob.glob(os.path.join(r_dir, "*.*")) if r_dir else []
            f_files = glob.glob(os.path.join(f_dir, "*.*")) if f_dir else []

            valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
            r_files = [f for f in r_files if Path(f).suffix.lower() in valid_exts][:n_samples]
            f_files = [f for f in f_files if Path(f).suffix.lower() in valid_exts][:n_samples]

            rows = []
            for f in r_files:
                rows.append({"frame_path": f, "video_id": Path(f).stem, "label": 0, "label_name": "real"})
            for f in f_files:
                rows.append({"frame_path": f, "video_id": Path(f).stem, "label": 1, "label_name": "fake"})

            df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
            _save_df(df, out_dir, split_name)

    else:
        print("[splits] Flat real/fake structure detected — creating splits automatically...")
        all_real: list[str] = []
        all_fake: list[str] = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

        for rd in real_dirs:
            all_real.extend([f for f in glob.glob(os.path.join(rd, "*.*")) if Path(f).suffix.lower() in valid_exts])
        for fd in fake_dirs:
            all_fake.extend([f for f in glob.glob(os.path.join(fd, "*.*")) if Path(f).suffix.lower() in valid_exts])

        random.shuffle(all_real)
        random.shuffle(all_fake)

        total_train = train_per_class
        total_val = val_per_class
        total_test = test_per_class

        slices = [
            ("train", 0, total_train),
            ("val", total_train, total_train + total_val),
            ("test", total_train + total_val, total_train + total_val + total_test),
        ]
        for split_name, start, end in slices:
            r_sub = all_real[start:end]
            f_sub = all_fake[start:end]
            rows = []
            for f in r_sub:
                rows.append({"frame_path": f, "video_id": Path(f).stem, "label": 0, "label_name": "real"})
            for f in f_sub:
                rows.append({"frame_path": f, "video_id": Path(f).stem, "label": 1, "label_name": "fake"})

            df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
            _save_df(df, out_dir, split_name)

    print("\n[splits] Done! Splits ready for training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input", help="Root directory to search for images")
    parser.add_argument("--out-dir", default="data/splits", help="Output directory for CSVs")
    parser.add_argument("--train-per-class", type=int, default=5000)
    parser.add_argument("--val-per-class", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=1000)
    args = parser.parse_args()

    auto_build_splits(
        input_root=args.input_root,
        out_dir=args.out_dir,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
    )
