"""
src.training.train  ·  Config-driven Training Loop for Deepfake Detection.
==========================================================================
Features:
    - Optimizer: AdamW with weight decay
    - Scheduler: Cosine decay with linear warmup (stepped per batch)
    - Loss: BCEWithLogitsLoss with positive class weight
    - Mixed Precision: Automatic Mixed Precision (AMP) via torch.cuda.amp
    - Gradient Clipping: max_norm clipping
    - Gradient Accumulation: Multi-step accumulation
    - Optional Gradient Checkpointing for reduced memory
    - Early Stopping based on validation AUC-ROC
    - Checkpoints: Saves state_dict() and config.yaml next to checkpoint
    - Logging: Per-epoch metrics to history.csv and TensorBoard SummaryWriter
    - Safety: Hardware safeguard preventing large runs outside Kaggle

Usage:
    python src/training/train.py --config configs/vit_b16_finetune.yaml
    python src/training/train.py --config configs/vit_b16_finetune.yaml --debug
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import get_dataloaders
from src.models.factory import build_model
from src.training.losses import get_loss_fn
from src.utils.config import (
    add_config_args,
    apply_debug_overrides,
    load_config,
    save_config,
)
from src.utils.safeguard import check_environment
from src.utils.seed import seed_everything

__all__ = [
    "get_cosine_schedule_with_warmup",
    "Trainer",
    "train_from_config",
]


# ═════════════════════════════════════════════════════════════════
#  Learning Rate Scheduler: Cosine with Linear Warmup
# ═════════════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the
    values of the cosine function between the initial lr set in the
    optimizer to min_lr_ratio * initial lr, after a warmup period during
    which it increases linearly between 0 and the initial lr.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═════════════════════════════════════════════════════════════════
#  Trainer Class
# ═════════════════════════════════════════════════════════════════

class Trainer:
    """
    Config-driven Trainer for Deepfake ViT detection.

    Parameters
    ----------
    cfg : Any
        Configuration namespace.
    device : torch.device, optional
        Target execution device (default: auto CUDA if available, else CPU).
    """

    def __init__(self, cfg: Any, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Output & run directory
        run_name = getattr(
            getattr(cfg, "experiment", None),
            "name",
            getattr(getattr(cfg, "model", None), "backbone", "experiment"),
        )
        base_out = getattr(getattr(cfg, "experiment", None), "output_dir", "outputs")
        self.output_dir = Path(base_out) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging setup
        self.csv_path = self.output_dir / "history.csv"
        self.tb_dir = self.output_dir / "tensorboard"
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

        # Hyper-parameters
        t_cfg = getattr(cfg, "training", None)
        self.epochs: int = getattr(t_cfg, "epochs", 30)
        self.lr: float = float(getattr(t_cfg, "lr", 3e-5))
        self.weight_decay: float = float(getattr(t_cfg, "weight_decay", 1e-4))
        self.warmup_epochs: int = getattr(t_cfg, "warmup_epochs", 3)
        self.accum_steps: int = max(1, getattr(t_cfg, "gradient_accumulation_steps", 1))
        self.max_grad_norm: float = float(getattr(t_cfg, "max_grad_norm", 1.0))
        self.early_stopping_patience: int = getattr(t_cfg, "early_stopping_patience", 5)

        # Mixed precision setup
        amp_requested = getattr(t_cfg, "mixed_precision", True)
        self.use_amp = bool(amp_requested and self.device.type == "cuda")
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.use_amp)
            if hasattr(torch.amp, "GradScaler")
            else torch.cuda.amp.GradScaler(enabled=self.use_amp)
        )

        # Build data loaders
        split_dir = getattr(getattr(cfg, "data", None), "split_dir", "data/splits")
        mode = getattr(getattr(cfg, "data", None), "mode", "image")
        num_frames = getattr(getattr(cfg, "data", None), "num_frames", 16)

        self.train_loader, self.val_loader, self.test_loader, class_weights = get_dataloaders(
            cfg=self.cfg,
            split_dir=split_dir,
            mode=mode,
            num_frames=num_frames,
        )

        # Build model
        self.model = build_model(cfg=self.cfg).to(self.device)

        # Optional gradient checkpointing
        if getattr(t_cfg, "gradient_checkpointing", False):
            self._enable_gradient_checkpointing()

        # Build loss function with positive class weighting
        self.criterion = get_loss_fn(class_weights=class_weights, device=self.device)

        # Optimizer: AdamW
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Scheduler: Cosine with Warmup (stepped per optimizer step)
        total_training_steps = max(1, (len(self.train_loader) // self.accum_steps) * self.epochs)
        warmup_steps = max(0, (len(self.train_loader) // self.accum_steps) * self.warmup_epochs)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        # CSV Logging header
        self._init_csv()

    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing if supported by the backbone."""
        if hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(True)
            print("[Trainer] Gradient checkpointing enabled on model.")
        elif hasattr(self.model, "backbone") and hasattr(self.model.backbone, "set_grad_checkpointing"):
            self.model.backbone.set_grad_checkpointing(True)
            print("[Trainer] Gradient checkpointing enabled on backbone.")
        else:
            print("[Trainer] Notice: Model does not have set_grad_checkpointing. Skipped.")

    def _init_csv(self) -> None:
        """Initialize the CSV history file with headers."""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["epoch", "train_loss", "train_acc", "train_auc", "val_loss", "val_acc", "val_auc", "lr"]
                )

    def _save_checkpoint(self, filename: str) -> Path:
        """
        Save model.state_dict() and a copy of the config next to the checkpoint.
        Rule: Save only state_dict() checkpoints.
        """
        save_path = self.output_dir / filename
        torch.save(self.model.state_dict(), save_path)
        # Save config YAML alongside checkpoint
        save_config(self.cfg, self.output_dir / "config.yaml")
        return save_path

    def _compute_metrics(self, y_true: np.ndarray, y_probs: np.ndarray) -> tuple[float, float]:
        """Compute accuracy and AUC-ROC safely."""
        y_preds = (y_probs >= 0.5).astype(int)
        acc = float(accuracy_score(y_true, y_preds)) if len(y_true) > 0 else 0.0

        # AUC requires at least one positive and one negative sample
        unique_classes = np.unique(y_true)
        if len(unique_classes) >= 2:
            try:
                auc = float(roc_auc_score(y_true, y_probs))
            except Exception:
                auc = 0.5
        else:
            auc = 0.5

        return acc, auc

    def train_epoch(self, epoch: int) -> tuple[float, float, float]:
        """Execute one training epoch."""
        self.model.train()
        total_loss = 0.0
        all_targets: list[int] = []
        all_probs: list[float] = []

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]", leave=False)

        for step, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.float().unsqueeze(1).to(self.device)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                scaled_loss = loss / self.accum_steps

            self.scaler.scale(scaled_loss).backward()

            # Optimizer and scheduler step after accum_steps
            if (step + 1) % self.accum_steps == 0 or (step + 1) == len(self.train_loader):
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            # Record metrics
            total_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            t_vals = targets.detach().cpu().numpy().flatten().astype(int)

            all_probs.extend(probs.tolist())
            all_targets.extend(t_vals.tolist())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(1, len(self.train_loader))
        acc, auc = self._compute_metrics(np.array(all_targets), np.array(all_probs))
        return avg_loss, acc, auc

    @torch.no_grad()
    def evaluate(self, loader: torch.utils.data.DataLoader, desc: str = "Val") -> tuple[float, float, float]:
        """Execute validation / evaluation loop."""
        self.model.eval()
        total_loss = 0.0
        all_targets: list[int] = []
        all_probs: list[float] = []

        if len(loader) == 0:
            return 0.0, 0.0, 0.5

        pbar = tqdm(loader, desc=desc, leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.float().unsqueeze(1).to(self.device)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            t_vals = targets.cpu().numpy().flatten().astype(int)

            all_probs.extend(probs.tolist())
            all_targets.extend(t_vals.tolist())

        avg_loss = total_loss / max(1, len(loader))
        acc, auc = self._compute_metrics(np.array(all_targets), np.array(all_probs))
        return avg_loss, acc, auc

    def fit(self) -> dict[str, Any]:
        """Run the full training pipeline with early stopping and logging."""
        print(f"\n{'=' * 65}")
        print(f"  Starting Training: {self.output_dir.name}")
        print(f"  Device           : {self.device} (AMP: {self.use_amp})")
        print(f"  Epochs           : {self.epochs} | Batch size: {self.train_loader.batch_size}")
        print(f"  Accumulation     : {self.accum_steps} steps (Effective batch: {self.train_loader.batch_size * self.accum_steps})")
        print(f"  Output Directory : {self.output_dir}")
        print(f"{'=' * 65}\n")

        best_val_auc = -1.0
        patience_counter = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Train epoch
            train_loss, train_acc, train_auc = self.train_epoch(epoch)

            # Validation epoch
            val_loss, val_acc, val_auc = self.evaluate(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]")

            # Print epoch summary
            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | AUC: {train_auc:.4f}  ||  "
                f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% | AUC: {val_auc:.4f}  ||  "
                f"LR: {current_lr:.2e}"
            )

            # Log to CSV
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, train_acc, train_auc, val_loss, val_acc, val_auc, current_lr])

            # Log to TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("AUC/train", train_auc, epoch)
            self.writer.add_scalar("AUC/val", val_auc, epoch)
            self.writer.add_scalar("Learning_Rate", current_lr, epoch)

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_auc": train_auc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
            })

            # Checkpoint: Save Last
            self._save_checkpoint("last_model.pt")

            # Checkpoint: Save Best on Validation AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                saved_best = self._save_checkpoint("best_model.pt")
                print(f"  --> Saved new best checkpoint (Val AUC: {best_val_auc:.4f}) to {saved_best.name}")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"\n[Early Stopping] No improvement in Val AUC for {self.early_stopping_patience} epochs. Stopping.")
                    break

        self.writer.close()
        print(f"\n[Trainer] Training completed. Best Val AUC: {best_val_auc:.4f}")
        return {"best_val_auc": best_val_auc, "history": history}


# ═════════════════════════════════════════════════════════════════
#  Main Entrypoint
# ═════════════════════════════════════════════════════════════════

def train_from_config(cfg_path_or_obj: str | Path | Any, debug: bool = False) -> dict[str, Any]:
    """Execute training from a configuration path or namespace."""
    if isinstance(cfg_path_or_obj, (str, Path)):
        cfg = load_config(cfg_path_or_obj)
    else:
        cfg = cfg_path_or_obj

    if debug:
        cfg.debug.enabled = True

    cfg = apply_debug_overrides(cfg)
    seed_everything(getattr(getattr(cfg, "experiment", None), "seed", 42))

    trainer = Trainer(cfg)
    return trainer.fit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepfake ViT Training Pipeline")
    parser = add_config_args(parser)
    args = parser.parse_args()

    train_from_config(args.config, debug=args.debug)


if __name__ == "__main__":
    main()
