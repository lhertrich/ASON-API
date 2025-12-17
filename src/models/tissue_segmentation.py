import torch
import torch.nn as nn
import functools
import wandb

from torchmetrics import Accuracy, F1Score
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Optional

from src.types import Batch


class TissueSegmentationModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        preprocessor: Optional[functools.partial] = None,
    ):
        super().__init__()
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.model = model.to(self.device)
        self.preprocessor = preprocessor
        self.cfg = cfg

        self.criterion = nn.CrossEntropyLoss()

        self.use_wandb = self.cfg.wandb.get("use", False)
        if self.use_wandb:
            wandb.init(
                project=cfg.wandb.get("project", "rp-tissue-segmentation"),
                name=cfg.model.get("name", None),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            wandb.watch(self.model, log="all", log_freq=50)

        num_classes = cfg.model.params.classes
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(
            self.device
        )
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(
            self.device
        )

        self.train_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average=cfg.eval.f1_avg
        ).to(self.device)

        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average=cfg.eval.f1_avg
        ).to(self.device)

        self.metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "train_f1": [],
            "test_loss": [],
            "test_accuracy": [],
            "test_f1": [],
        }

        self.current_epoch = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Batch) -> torch.Tensor:
        """Performs a training step for the model.

        Args:
            batch (Batch): A batch of images and masks.

        Returns:
            torch.Tensor: The loss for the batch.
        """
        image, mask = batch
        if self.preprocessor is not None:
            x = self.preprocessor(image)
        else:
            x = image

        y = mask.squeeze(1) if mask.dim() > 3 else mask

        logits = self.model(x)
        loss = self.criterion(logits, y)

        self.train_accuracy.update(logits, y)
        self.train_f1.update(logits, y)

        return loss, loss.item()

    def training_epoch_end(self, avg_train_loss: float) -> Dict[str, float]:
        """Finalizes the training epoch and returns the metrics.

        Returns:
            Dict[str, float]: The metrics for the training epoch.
        """
        metrics = {
            "train_loss": avg_train_loss,
            "train_accuracy": self.train_accuracy.compute().item(),
            "train_f1": self.train_f1.compute().item(),
            "epoch": self.current_epoch,
        }

        self.metrics["train_loss"].append(metrics["train_loss"])
        self.metrics["train_accuracy"].append(metrics["train_accuracy"])
        self.metrics["train_f1"].append(metrics["train_f1"])

        if self.use_wandb:
            wandb.log(
                {
                    "train/loss": metrics["train_loss"],
                    "train/accuracy": metrics["train_accuracy"],
                    "train/f1": metrics["train_f1"],
                    "epoch": self.current_epoch,
                }
            )

        self.train_accuracy.reset()
        self.train_f1.reset()

        self.current_epoch += 1

        return metrics

    def test_step(self, batch: Batch) -> Dict[str, float]:
        """Performs a test step for the model.

        Args:
            batch (Batch): A batch of images and masks.

        Returns:
            Dict[str, float]: The metrics for the test epoch.
        """
        image, mask = batch
        if self.preprocessor is not None:
            x = self.preprocessor(image)
        else:
            x = image

        y = mask.squeeze(1) if mask.dim() > 3 else mask

        with torch.no_grad():
            logits = self.model(x)
            loss = self.criterion(logits, y)

        self.test_accuracy.update(logits, y)
        self.test_f1.update(logits, y)

        return loss.item()

    def test_epoch_end(self, avg_test_loss: float) -> Dict[str, float]:
        """Finalizes the test epoch and returns the metrics.

        Returns:
            Dict[str, float]: The metrics for the test epoch.
        """
        metrics = {
            "test_loss": avg_test_loss,
            "test_accuracy": self.test_accuracy.compute().item(),
            "test_f1": self.test_f1.compute().item(),
        }

        self.metrics["test_loss"].append(metrics["test_loss"])
        self.metrics["test_accuracy"].append(metrics["test_accuracy"])
        self.metrics["test_f1"].append(metrics["test_f1"])

        if self.use_wandb:
            wandb.log(
                {
                    "test/loss": metrics["test_loss"],
                    "test/accuracy": metrics["test_accuracy"],
                    "test/f1": metrics["test_f1"],
                }
            )

        self.test_accuracy.reset()
        self.test_f1.reset()

        return metrics

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.model.training.learning_rate,
            weight_decay=self.cfg.model.training.weight_decay,
        )
        return optimizer

    def finish(self):
        if self.use_wandb:
            wandb.finish()
