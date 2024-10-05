import os
import pandas as pd
import torch
import psutil
import time

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from PIL import Image, ImageOps


WORKER_COUNT = 2


class FEMNIST(Dataset):

    def __init__(self, stage="train", transform=None, data_dist="local", client_id=0, *args, **kwargs):
        super(FEMNIST, self).__init__()
        self.file_path = os.path.dirname(os.path.realpath(__file__))

        self.client_id = 0 if data_dist == "local" else client_id
        self.stage = stage

        self.transform = transform
        self.indices = pd.read_csv(f"{self.file_path}/data/processed/{data_dist}/client_{client_id}.csv")
        self.indices = self.indices.loc[self.indices["fold"] == stage]
        self.indices = self.indices.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.indices.iloc[idx]
        label = item["class_id"]

        try:
            image = Image.open(f"{self.file_path}/data/{item['path']}")
            image = ImageOps.grayscale(image)

            if self.transform is not None:
                image = self.transform(image)

            image = torch.Tensor(image)
            return image, label
        except FileNotFoundError:
            pass


class FEMNISTLightningData(pl.LightningDataModule):

    def __init__(self, data_dist="local", client_id=0, batch_size=32, num_workers=4, *args, **kwargs):
        super(FEMNISTLightningData, self).__init__()
        self.batch_size = batch_size
        self.data_dist = data_dist
        self.client_id = client_id
        self.num_workers = num_workers

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        self.val_test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # We need the following for DP workloads, even though it's duplicate code
        self.dataset = FEMNIST(stage="train", transform=self.train_transform, data_dist=self.data_dist,
                               client_id=self.client_id)
        self.len_trainset = len(self.dataset)

    def train_dataloader(self):
        self.dataset = FEMNIST(stage="train", transform=self.train_transform, data_dist=self.data_dist,
                       client_id=self.client_id)
        self.len_trainset = len(self.dataset)
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=WORKER_COUNT)

    def val_dataloader(self):
        self.dataset = FEMNIST(stage="val", transform=self.val_test_transform, data_dist=self.data_dist,
                       client_id=self.client_id)
        self.len_valset = len(self.dataset)
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=WORKER_COUNT)

    def test_dataloader(self):
        self.dataset = FEMNIST(stage="test", transform=self.val_test_transform, data_dist=self.data_dist,
                       client_id=self.client_id)
        self.len_testset = len(self.dataset)
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=WORKER_COUNT)

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        self.batch_move_start_time = time.time()
        super(FEMNISTLightningData, self).on_before_batch_transfer(batch, dataloader_idx)
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        super(FEMNISTLightningData, self).on_after_batch_transfer(batch, dataloader_idx)
        try:
            self.logger.experiment.log({"timing/train/batch_move_time": time.time() - self.batch_move_start_time})
        except AttributeError:
            pass

        return batch


if __name__ == "__main__":
    loader = FEMNIST()
    print(next(iter(loader)))
