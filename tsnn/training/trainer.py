"""Main TSNN trainer orchestrating the 3-stage training pipeline.

Stage A: Self-supervised MD pretraining (MISATO, MDbind, MDD)
Stage B: Dissociation-aware pretraining (DD-13M)
Stage C: Kinetics fine-tuning (BindingDB, KOFFI, PDBbind-k_off)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from tsnn.model.tsnn import TSNN, TSNNConfig
from tsnn.utils.io import save_checkpoint, load_checkpoint, count_parameters

logger = logging.getLogger(__name__)


class TSNNTrainer:
    """Orchestrates the three-stage TSNN training pipeline.

    Args:
        config: TSNNConfig for model construction.
        output_dir: Directory for checkpoints and logs.
        device: Training device.
        lr: Base learning rate.
        weight_decay: Weight decay for AdamW.
        grad_clip: Maximum gradient norm.
        mixed_precision: Use AMP for faster training.
    """

    def __init__(
        self,
        config: TSNNConfig | None = None,
        output_dir: str = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        mixed_precision: bool = True,
    ):
        self.config = config or TSNNConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision and device != "cpu"

        # Build model
        self.model = TSNN(self.config).to(device)
        logger.info(f"Model parameters: {count_parameters(self.model):,}")

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda") if self.mixed_precision else None

    def _build_optimizer(self, lr: float | None = None) -> AdamW:
        return AdamW(
            self.model.parameters(),
            lr=lr or self.lr,
            weight_decay=self.weight_decay,
        )

    def _build_scheduler(
        self, optimizer, num_epochs: int, warmup_epochs: int = 5
    ):
        warmup = LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs
        )
        return SequentialLR(optimizer, [warmup, cosine],
                            milestones=[warmup_epochs])

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: AdamW,
        loss_fn,
        epoch: int,
    ) -> dict[str, float]:
        """Run a single training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            if self.mixed_precision:
                with torch.amp.autocast("cuda"):
                    loss_dict = self._compute_loss(batch, loss_fn)
                    loss = loss_dict["total"]
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss_dict = self._compute_loss(batch, loss_fn)
                loss = loss_dict["total"]
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 50 == 0:
                logger.info(
                    f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                    f"loss={loss.item():.4f}"
                )

        return {"train_loss": total_loss / max(n_batches, 1)}

    def _compute_loss(self, batch, loss_fn):
        """Compute loss for a batch — handles both pretraining and fine-tuning."""
        frames = batch["frames"]
        labels = batch["labels"]

        # Build model inputs
        frame_dicts = []
        cross_masks = []
        e2c_list = []

        for frame in frames:
            fd = {
                "node_features": frame.x.to(self.device),
                "positions": frame.pos.to(self.device),
                "edge_index": frame.edge_index.to(self.device),
                "edge_attr": (
                    frame.edge_attr.to(self.device)
                    if frame.edge_attr is not None else None
                ),
            }
            frame_dicts.append(fd)
            cross_masks.append(frame.cross_edge_mask.to(self.device))
            e2c_list.append(
                torch.zeros(frame.edge_index.shape[1], dtype=torch.long,
                            device=self.device)
            )

        n2c = torch.zeros(frames[0].num_nodes, dtype=torch.long,
                          device=self.device)

        output = self.model(frame_dicts, cross_masks, n2c, e2c_list,
                            num_complexes=1)

        targets = {}
        if labels.get("koff") is not None:
            targets["log_koff"] = torch.tensor(
                [labels["koff"]], device=self.device, dtype=torch.float32
            )
        else:
            targets["log_koff"] = torch.tensor(
                [float("nan")], device=self.device
            )

        if labels.get("censored") is not None:
            targets["censored"] = torch.tensor(
                [labels["censored"]], device=self.device, dtype=torch.bool
            )
        else:
            targets["censored"] = torch.tensor([False], device=self.device)

        T = len(frames)
        targets["event_times"] = torch.tensor(
            [T - 1], device=self.device, dtype=torch.long
        )

        return loss_fn(output, targets)

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader, loss_fn) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            loss_dict = self._compute_loss(batch, loss_fn)
            total_loss += loss_dict["total"].item()
            n_batches += 1

        return {"val_loss": total_loss / max(n_batches, 1)}

    def run_stage_a(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        loss_fn=None,
        num_epochs: int = 50,
        lr: float = 3e-4,
    ) -> None:
        """Stage A: Self-supervised MD pretraining."""
        logger.info("=" * 60)
        logger.info("STAGE A: Self-supervised MD pretraining")
        logger.info("=" * 60)

        if loss_fn is None:
            from tsnn.losses.pretraining_losses import StageALoss
            loss_fn = StageALoss()

        optimizer = self._build_optimizer(lr)
        scheduler = self._build_scheduler(optimizer, num_epochs)

        best_val_loss = float("inf")
        for epoch in range(1, num_epochs + 1):
            train_metrics = self._train_epoch(train_loader, optimizer,
                                               loss_fn, epoch)
            scheduler.step()

            if val_loader is not None:
                val_metrics = self._validate(val_loader, loss_fn)
                metrics = {**train_metrics, **val_metrics}

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    save_checkpoint(
                        self.model, optimizer, epoch, metrics,
                        self.output_dir / "stage_a_best.pt",
                    )
            else:
                metrics = train_metrics

            logger.info(f"Epoch {epoch}/{num_epochs}: {metrics}")

        save_checkpoint(
            self.model, optimizer, num_epochs, metrics,
            self.output_dir / "stage_a_final.pt",
        )
        logger.info("Stage A complete.")

    def run_stage_b(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        loss_fn=None,
        num_epochs: int = 30,
        lr: float = 1e-4,
        stage_a_checkpoint: str | None = None,
    ) -> None:
        """Stage B: Dissociation-aware pretraining on DD-13M."""
        logger.info("=" * 60)
        logger.info("STAGE B: Dissociation pretraining (DD-13M)")
        logger.info("=" * 60)

        # Load Stage A weights
        if stage_a_checkpoint:
            load_checkpoint(stage_a_checkpoint, self.model)
        elif (self.output_dir / "stage_a_best.pt").exists():
            load_checkpoint(self.output_dir / "stage_a_best.pt", self.model)

        if loss_fn is None:
            from tsnn.losses.pretraining_losses import StageBLoss
            loss_fn = StageBLoss()

        optimizer = self._build_optimizer(lr)
        scheduler = self._build_scheduler(optimizer, num_epochs, warmup_epochs=3)

        best_val_loss = float("inf")
        for epoch in range(1, num_epochs + 1):
            train_metrics = self._train_epoch(train_loader, optimizer,
                                               loss_fn, epoch)
            scheduler.step()

            if val_loader is not None:
                val_metrics = self._validate(val_loader, loss_fn)
                metrics = {**train_metrics, **val_metrics}
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    save_checkpoint(
                        self.model, optimizer, epoch, metrics,
                        self.output_dir / "stage_b_best.pt",
                    )
            else:
                metrics = train_metrics

            logger.info(f"Epoch {epoch}/{num_epochs}: {metrics}")

        save_checkpoint(
            self.model, optimizer, num_epochs, metrics,
            self.output_dir / "stage_b_final.pt",
        )
        logger.info("Stage B complete.")

    def run_stage_c(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        loss_fn=None,
        num_epochs: int = 100,
        lr: float = 5e-5,
        stage_b_checkpoint: str | None = None,
    ) -> None:
        """Stage C: Kinetics fine-tuning."""
        logger.info("=" * 60)
        logger.info("STAGE C: Kinetics fine-tuning")
        logger.info("=" * 60)

        # Load Stage B weights
        if stage_b_checkpoint:
            load_checkpoint(stage_b_checkpoint, self.model)
        elif (self.output_dir / "stage_b_best.pt").exists():
            load_checkpoint(self.output_dir / "stage_b_best.pt", self.model)
        elif (self.output_dir / "stage_a_best.pt").exists():
            load_checkpoint(self.output_dir / "stage_a_best.pt", self.model)

        if loss_fn is None:
            from tsnn.losses.combined import CombinedLoss
            loss_fn = CombinedLoss()

        # Discriminative learning rate: lower for pretrained, higher for heads
        pretrained_params = list(self.model.encoder.parameters()) + \
                           list(self.model.sheaf_transport.parameters())
        head_params = list(self.model.hazard_head.parameters()) + \
                     list(self.model.survival_head.parameters())

        optimizer = AdamW([
            {"params": pretrained_params, "lr": lr * 0.1},
            {"params": head_params, "lr": lr},
        ], weight_decay=self.weight_decay)

        scheduler = self._build_scheduler(optimizer, num_epochs, warmup_epochs=10)

        best_val_loss = float("inf")
        for epoch in range(1, num_epochs + 1):
            train_metrics = self._train_epoch(train_loader, optimizer,
                                               loss_fn, epoch)
            scheduler.step()

            if val_loader is not None:
                val_metrics = self._validate(val_loader, loss_fn)
                metrics = {**train_metrics, **val_metrics}
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    save_checkpoint(
                        self.model, optimizer, epoch, metrics,
                        self.output_dir / "stage_c_best.pt",
                    )
            else:
                metrics = train_metrics

            logger.info(f"Epoch {epoch}/{num_epochs}: {metrics}")

        save_checkpoint(
            self.model, optimizer, num_epochs, metrics,
            self.output_dir / "stage_c_final.pt",
        )
        logger.info("Stage C complete. Model ready for evaluation.")
