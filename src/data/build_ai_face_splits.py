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
    if df.empty or "label" not in df.columns:
        print(f"  [WARNING] No data found for split '{split_name}'")
        df = pd.DataFrame(columns=["frame_path", "video_id", "label", "label_name"])
    df.to_csv(out_csv, index=False)
    real_cnt = (df["label"] == 0).sum() if not df.empty else 0
    fake_cnt = (df["label"] == 1).sum() if not df.empty else 0
    print(f"  Saved {out_csv}: {len(df):,} items (Real: {real_cnt:,}, Fake: {fake_cnt:,})")

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

    # Check if dedicated train AND (val or test) folders exist
    has_explicit_splits = (
        any("train" in r.lower() for r in real_dirs)
        and (
            any("val" in r.lower() or "valid" in r.lower() for r in real_dirs)
            or any("test" in r.lower() for r in real_dirs)
        )
    )

    if has_explicit_splits:
        print("[splits] Pre-partitioned train/val/test folders detected!")
        split_configs = [
            ("train", ["train"], train_per_class),
            ("val", ["val", "valid"], val_per_class),
            ("test", ["test"], test_per_class),
        ]
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        for split_name, keywords, n_samples in split_configs:
            r_dir = next((d for d in real_dirs if any(k in d.lower() for k in keywords)), None)
            f_dir = next((d for d in fake_dirs if any(k in d.lower() for k in keywords)), None)

            r_files = glob.glob(os.path.join(r_dir, "*.*")) if r_dir else []
            f_files = glob.glob(os.path.join(f_dir, "*.*")) if f_dir else []

            r_files = [f for f in r_files if Path(f).suffix.lower() in valid_exts][:n_samples]
            f_files = [f for f in f_files if Path(f).suffix.lower() in valid_exts][:n_samples]

            rows = []
            for f in r_files:
                rows.append({"frame_path": f, "video_id": Path(f).stem, "label": 0, "label_name": "real"})
            for f in f_files:
                rows.append({"frame_path": f, "video_id": Path(f).stem, "label": 1, "label_name": "fake"})

            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            _save_df(df, out_dir, split_name)

    else:
        print("[splits] Gathering all images and creating balanced 70% / 15% / 15% splits...")
        all_real: list[str] = []
        all_fake: list[str] = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

        for rd in real_dirs:
            all_real.extend([f for f in glob.glob(os.path.join(rd, "*.*")) if Path(f).suffix.lower() in valid_exts])
        for fd in fake_dirs:
            all_fake.extend([f for f in glob.glob(os.path.join(fd, "*.*")) if Path(f).suffix.lower() in valid_exts])

        # Remove duplicate paths if any
        all_real = sorted(list(set(all_real)))
        all_fake = sorted(list(set(all_fake)))

        random.shuffle(all_real)
        random.shuffle(all_fake)

        print(f"[splits] Total unique images collected: {len(all_real):,} Real, {len(all_fake):,} Fake")

        n_r = len(all_real)
        n_f = len(all_fake)

        n_r_train = min(train_per_class, max(1, int(n_r * 0.70)))
        n_r_val = min(val_per_class, max(1, int(n_r * 0.15)))
        n_r_test = min(test_per_class, max(1, n_r - n_r_train - n_r_val))

        n_f_train = min(train_per_class, max(1, int(n_f * 0.70)))
        n_f_val = min(val_per_class, max(1, int(n_f * 0.15)))
        n_f_test = min(test_per_class, max(1, n_f - n_f_train - n_f_val))

        splits_data = [
            ("train", all_real[:n_r_train], all_fake[:n_f_train]),
            ("val", all_real[n_r_train:n_r_train + n_r_val], all_fake[n_f_train:n_f_train + n_f_val]),
            ("test", all_real[n_r_train + n_r_val:n_r_train + n_r_val + n_r_test],
                     all_fake[n_f_train + n_f_val:n_f_train + n_f_val + n_f_test]),
        ]

        for split_name, r_sub, f_sub in splits_data:
            rows = []
            for f in r_sub:
                rows.append({"frame_path": f, "video_id": Path(f).stem, "label": 0, "label_name": "real"})
            for f in f_sub:
                rows.append({"frame_path": f, "video_id": Path(f).stem, "label": 1, "label_name": "fake"})

            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
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
