import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy


class LightningModule(pl.LightningModule):
    def __init__(
        self,
        model: "torch.nn.Module",
    ):
        super().__init__()
        self.model = model

        self.train_accuracy = BinaryAccuracy()
        self.valid_accuracy = BinaryAccuracy()

    def criterion(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x).flatten()

        loss = self.criterion(y_hat, y)
        self.train_accuracy(y_hat, y)

        return {
            "loss": loss,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_hat = self.model(x).flatten()

        loss = self.criterion(y_hat, y)
        self.valid_accuracy(y_hat, y)

        return {
            "loss": loss,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch

        y_hat = self.model(x).flatten()

        return y_hat

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_accuracy.compute(), prog_bar=True)
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        self.log("valid_acc", self.valid_accuracy.compute(), prog_bar=True)
        self.valid_accuracy.reset()

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
