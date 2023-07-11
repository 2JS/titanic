import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from data import TitanicDataset
from lightning_utils.loggers import OutputLogger
from lightningmodule import LightningModule
from model import SimpleModel


def main():
    pl.seed_everything(528491)

    model = SimpleModel()
    lightning_module = LightningModule(model)

    dataset = TitanicDataset("data/train.csv")
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, lengths=[0.9, 0.1]
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=32, shuffle=False
    )

    logger = WandbLogger(project="titanic")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[OutputLogger()],
        max_epochs=100,
    )

    trainer.fit(
        lightning_module,
        train_dataloader,
        valid_dataloader,
    )


if __name__ == "__main__":
    main()
