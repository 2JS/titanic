import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from data import TitanicDataset
from lightningmodule import LightningModule
from model import SimpleModel


def main():
    trainer = pl.Trainer(accelerator="gpu")

    dataset = TitanicDataset("data/test.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = SimpleModel()
    lightningmodule = LightningModule(model)

    y_hat = trainer.predict(
        lightningmodule,
        dataloader,
        return_predictions=True,
        ckpt_path="path/to/checkpoint.ckpt",
    )

    y_hat = torch.cat(y_hat)

    survived = torch.where(y_hat > 0, 1, 0)

    with open("submission.csv", "w") as f:
        print("PassengerId,Survived", file=f)
        for i, s in zip(range(892, 1310), survived):
            print(i, s.item(), file=f, sep=",")


if __name__ == "__main__":
    main()
