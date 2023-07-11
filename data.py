import pandas as pd
import torch
from torch.utils.data import Dataset


class TitanicDataset(Dataset):
    def __init__(
        self,
        path: str,
    ):
        super().__init__()
        self.path = path

        dataframe = pd.read_csv(self.path)

        self.x, self.y = self._preprocess(dataframe)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    @staticmethod
    def _preprocess(dataframe):
        x = dataframe[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
        x = x.fillna(0)

        x["Sex"] = x["Sex"].map(lambda x: 0 if x == "male" else 1)
        x["Embarked"] = x["Embarked"].map(
            lambda x: 0 if x == "S" else 1 if x == "C" else 2
        )

        x = torch.tensor(x.values, dtype=torch.float32)

        min, _ = x.min(0)
        max, _ = x.max(0)
        x = (x - min) / (max - min)

        y = dataframe["Survived"]
        y = torch.tensor(y.values, dtype=torch.float32)

        return x, y


if __name__ == "__main__":
    dataset = TitanicDataset("data/train.csv")
    print(dataset[0])
    print(dataset[1])
