"""
Centralized baseline trainer for FL-IoT-IDS.

Contains:
- CentralizedTrainer: Standalone centralized trainer with validation tracking,
  early stopping, progress output, and Top-K multi-version checkpointing.

Adheres to:
- skills/model-design-and-implementation/SKILL.md
- skills/methodology-audit/SKILL.md (Pillar 2: Mathematical Correctness)
"""

import copy
import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object
    torch = None
    DataLoader = object

from src.base.trainer import BaseTrainer

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class CentralizedTrainer(BaseTrainer):
        """
        Standalone centralized baseline trainer (Scenario E1).
        Includes epoch loop, validation tracking, early stopping, and Top-K checkpointing.
        """
        def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: Union[str, torch.device] = "cpu",
            max_grad_norm: float = 5.0
        ):
            super().__init__(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                max_grad_norm=max_grad_norm
            )
            self.best_model_weights = None
            self.best_val_loss = float("inf")
            self.best_val_acc = 0.0
            self.top_k_checkpoints: List[Dict[str, Any]] = []

        def train_epoch(
            self,
            dataloader: DataLoader,
            verbose: bool = False,
            desc: str = "Training",
            **kwargs: Any
        ) -> Tuple[float, float]:
            """
            Executes a single epoch of centralized training.
            """
            self.model.train()
            t_loss, t_correct, t_total = 0.0, 0, 0

            iterator = dataloader
            if verbose:
                try:
                    from tqdm.auto import tqdm
                    iterator = tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True)
                except ImportError:
                    iterator = dataloader

            for X_b, y_b in iterator:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(X_b)
                loss = self.criterion(out, y_b)
                loss.backward()
                self.clip_gradients()
                self.optimizer.step()

                batch_samples = len(y_b)
                t_loss += loss.item() * batch_samples
                t_correct += (out.argmax(dim=1) == y_b).sum().item()
                t_total += batch_samples

                if verbose and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{(t_correct / max(t_total, 1)) * 100:.1f}%"
                    })

            train_loss = t_loss / max(t_total, 1)
            train_acc = t_correct / max(t_total, 1)
            return train_loss, train_acc

        def fit(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 20,
            patience: int = 5,
            monitor: str = "val_loss",
            save_top_k: int = 3,
            verbose: bool = True
        ) -> Dict[str, List[float]]:
            """
            Executes full centralized training with optional early stopping and live progress reporting.

            Args:
                train_loader: Training DataLoader.
                val_loader: Validation DataLoader.
                epochs: Total training epochs.
                patience: Early stopping patience.
                monitor: Metric to monitor for best checkpointing ('val_loss' or 'val_acc').
                save_top_k: Number of best model versions to track and preserve.
                verbose: Whether to print live progress to stdout.

            Returns:
                History dictionary containing 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
            """
            history: Dict[str, List[float]] = {
                "train_loss": [], "train_acc": [],
                "val_loss": [], "val_acc": []
            }
            patience_counter = 0

            for epoch in range(1, epochs + 1):
                # 1. Train epoch
                train_loss, train_acc = self.train_epoch(
                    train_loader,
                    verbose=verbose,
                    desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]"
                )
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)

                # 2. Validation step
                val_loss, val_acc = 0.0, 0.0
                is_best = ""
                early_stop = False

                if val_loader is not None:
                    val_loss, val_acc = self.evaluate(val_loader)
                    history["val_loss"].append(val_loss)
                    history["val_acc"].append(val_acc)

                    # Determine optimization direction ('max' for accuracy, 'min' for loss)
                    mode = "max" if monitor.lower() in ["val_acc", "accuracy", "acc"] else "min"
                    score = val_acc if mode == "max" else val_loss

                    # Check if current epoch qualifies for Top-K best models
                    qualifies_top_k = False
                    if len(self.top_k_checkpoints) < save_top_k:
                        qualifies_top_k = True
                    elif mode == "max" and score > self.top_k_checkpoints[-1]["score"]:
                        qualifies_top_k = True
                    elif mode == "min" and score < self.top_k_checkpoints[-1]["score"]:
                        qualifies_top_k = True

                    is_best = ""
                    if qualifies_top_k:
                        ckpt_entry = {
                            "epoch": epoch,
                            "score": float(score),
                            "val_loss": float(val_loss),
                            "val_acc": float(val_acc),
                            "train_loss": float(train_loss),
                            "train_acc": float(train_acc),
                            "state_dict": copy.deepcopy(self.model.state_dict())
                        }
                        self.top_k_checkpoints.append(ckpt_entry)
                        self.top_k_checkpoints.sort(key=lambda x: x["score"], reverse=(mode == "max"))
                        if len(self.top_k_checkpoints) > save_top_k:
                            self.top_k_checkpoints.pop()

                        # Determine rank of current epoch
                        rank = next(i + 1 for i, c in enumerate(self.top_k_checkpoints) if c["epoch"] == epoch)
                        if rank == 1:
                            self.best_model_weights = copy.deepcopy(self.model.state_dict())
                            self.best_val_loss = val_loss
                            self.best_val_acc = val_acc
                            patience_counter = 0
                            is_best = " ⭐ (Top-1 Best)"
                        else:
                            is_best = f" 🎖️ (Top-{rank})"
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            early_stop = True

                progress_msg = (
                    f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%{is_best}"
                )
                if verbose:
                    print(progress_msg)
                logger.info(progress_msg)

                if early_stop:
                    stop_msg = f"⏹️ Early stopping triggered at epoch {epoch} (patience={patience} on '{monitor}')"
                    if verbose:
                        print(stop_msg)
                    logger.info(stop_msg)
                    break

            # Restore best weights if available
            if self.best_model_weights is not None:
                self.model.load_state_dict(self.best_model_weights)
                best_summary = f"Loss: {self.best_val_loss:.4f}" if monitor.lower() not in ["val_acc", "accuracy", "acc"] else f"Acc: {self.best_val_acc*100:.2f}%"
                restore_msg = f"🏆 Restored model weights from best validation epoch (Monitored '{monitor}': {best_summary})"
                if verbose:
                    print(restore_msg)
                logger.info(restore_msg)

            return history

        def get_top_k_summary(self) -> List[Dict[str, Any]]:
            """Returns metadata summary of all preserved Top-K model checkpoints."""
            summary = []
            for rank, ckpt in enumerate(self.top_k_checkpoints, start=1):
                summary.append({
                    "rank": rank,
                    "epoch": ckpt["epoch"],
                    "score": round(ckpt["score"], 4),
                    "val_loss": round(ckpt["val_loss"], 4),
                    "val_acc": round(ckpt["val_acc"], 4),
                    "train_loss": round(ckpt["train_loss"], 4),
                    "train_acc": round(ckpt["train_acc"], 4),
                })
            return summary

        def save_top_k(
            self,
            output_dir: str = "checkpoints",
            prefix: str = "model",
            monitor: str = "val_loss"
        ) -> List[str]:
            """
            Serializes all preserved Top-K model checkpoints to disk and generates
            a manifest JSON tracking their ranks, epochs, and validation metrics.
            """
            os.makedirs(output_dir, exist_ok=True)
            saved_paths: List[str] = []

            for rank, ckpt in enumerate(self.top_k_checkpoints, start=1):
                metric_tag = "valacc" if monitor.lower() in ["val_acc", "accuracy", "acc"] else "valloss"
                filename = f"{prefix}_top{rank}_epoch{ckpt['epoch']:02d}_{metric_tag}_{ckpt['score']:.4f}.pth"
                filepath = os.path.join(output_dir, filename)
                torch.save(ckpt["state_dict"], filepath)
                saved_paths.append(filepath)

            # Also save canonical best model checkpoint (copy of rank 1)
            if self.top_k_checkpoints:
                best_path = os.path.join(output_dir, f"{prefix}_best.pth")
                torch.save(self.top_k_checkpoints[0]["state_dict"], best_path)
                saved_paths.append(best_path)

            manifest_path = os.path.join(output_dir, f"{prefix}_top_k_manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(self.get_top_k_summary(), f, indent=4)
            saved_paths.append(manifest_path)

            logger.info(f"Saved {len(self.top_k_checkpoints)} Top-K model versions to {output_dir}")
            return saved_paths

        def load_checkpoint_by_rank(self, rank: int = 1) -> None:
            """Loads model weights corresponding to a specific Top-K rank (1 to K)."""
            if not self.top_k_checkpoints:
                raise ValueError("No Top-K checkpoints have been recorded.")
            if not 1 <= rank <= len(self.top_k_checkpoints):
                raise IndexError(f"Rank {rank} out of range (available ranks: 1 to {len(self.top_k_checkpoints)})")
            target_ckpt = self.top_k_checkpoints[rank - 1]
            self.model.load_state_dict(target_ckpt["state_dict"])
            logger.info(f"Loaded model weights for Rank {rank} (Epoch {target_ckpt['epoch']}, Score: {target_ckpt['score']:.4f})")

else:
    class CentralizedTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for CentralizedTrainer. Please install torch.")
