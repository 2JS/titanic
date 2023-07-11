import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightningModule(pl.LightningModule):
    def __init__(
        self,
        model: "torch.nn.Module",
    ):
        super().__init__()
        self.model = model

    def criterion(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def common_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x).flatten()

        loss = self.criterion(y_hat, y)

        return {
            "loss": loss,
        }

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.common_step(batch, batch_idx)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
