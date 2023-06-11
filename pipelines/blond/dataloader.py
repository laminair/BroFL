import numpy as np
import psutil
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose

import json
import os
import pandas as pd
import h5py
import pytorch_lightning as pl
import time
from pathlib import Path

try:
    from pipelines.blond import utils
except ModuleNotFoundError:
    import utils
except OSError:
    pass


WORKER_COUNT = 2


class BLOND(Dataset):

    def __init__(self, transform=None, client_id=0, data_dist="local", stage="train", use_npy=False):
        """
        This initiates a PyTorch dataset for the BLOND dataset. It loads event snippets for classification.
        Since in some PyTorch versions we observed a compatibility issue with torchaudio and the power transformation
        procedure, we preprocessed the dataset and saved the input features in CSV files.
        Use "use_npy" to load the CSV files instead of the HDF5 files.
        """
        self.client_id = 0 if data_dist == "local" else client_id
        self.setting = data_dist
        self.use_npy = use_npy

        try:
            self.transform = Compose([utils.RandomAugment(p=0), utils.ACPower(), utils.MFCC()])
        except NameError:
            self.transform = None
            self.use_npy = True

        self.labels = []
        self.nan_keys = 0
        self.measurements = []

        self.file_path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.indices_base_path = Path(f"{self.file_path}/data/processed")
        self.data_base_path = Path(f"{self.file_path}/data/raw_data")

        self.indices = Path(f"{self.indices_base_path}/{data_dist}/client_{self.client_id}.csv")

        self.data = pd.read_csv(self.indices.absolute())
        self.data = self.data.loc[self.data["fold"] == stage]
        # Shuffle the dataset
        self.data = self.data.sample(frac=1.0).reset_index(drop=True)

        # Extract labels
        self.labels = self.data["Type"]

        with open(f"{self.file_path}/data/class_map.json", "r") as f:
            self.class_map = json.load(f)
            f.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        label = item["Type"]

        file_name = f"{item['Medal']}_{item['Socket']}_{item['Appliance']}_{item['Timestamp']}"

        if self.use_npy is False:
            data_file = Path(f"{self.data_base_path}/event_snippets/medal-{item['Medal']}/{file_name}.h5")
            features, label = self.load_hdf5_item(data_file=data_file, label=label)
        else:
            data_file = Path(f"{self.data_base_path}/event_snippets_npy/medal-{item['Medal']}/{file_name}.npy")
            features, label = self.load_csv_item(data_file=data_file, label=label)

        return features, label

    def load_csv_item(self, data_file: Path, label: str):
        try:
            features = np.load(f"{data_file.absolute()}")
            features_pt = torch.tensor(features)
            numeric_label = self.class_map[label]
            return features_pt.float(), numeric_label
        except FileNotFoundError:
            pass

    def load_hdf5_item(self, data_file: Path, label: str):
        try:
            f = h5py.File(data_file.absolute(), 'r')
            # Cut length of measurement window
            current = torch.as_tensor(f['data']['block0_values'][:, 1])
            voltage = torch.as_tensor(f['data']['block0_values'][:, 0])

            # Shifts event window to start with a new cycle
            idx = torch.where(torch.diff(torch.signbit(current[:1000])))[0][0]
            current = current[idx: 24576 + idx]
            voltage = voltage[idx: 24576 + idx]

            assert (len(current) == 24576)

            # Apply feature transform on current/voltage, if no
            # transform applied return (current, voltage, class)
            numeric_label = self.class_map[label]

            sample = (current, voltage, None, numeric_label)
            if self.transform:
                _, _, features, _ = self.transform(sample)
                return features.float(), numeric_label
            else:
                return sample[0].float(), sample[1].float(), sample[3]
        except FileNotFoundError:
            pass


class BlondLightningData(pl.LightningDataModule):

    def __init__(self, batch_size, data_dist="local", client_id=0, use_npy=True, *args, **kwargs):
        super(BlondLightningData, self).__init__()
        self.batch_size = batch_size
        self.data_dist = data_dist
        self.client_id = client_id
        self.use_npy = use_npy

        if self.client_id > 14 and self.data_dist == "natural":
            # We can only split by 15 medals (clients). Therefore, we opt to reuse subsets and assume almost identical
            # use patters, which may occur in an office setting. For IID settings we create splits to strictly fulfill
            # the IID criterion, no deviation.
            self.client_id = client_id % 14

        if self.use_npy is True:
            # IMPORTANT NOTE: We cannot apply the RandomAugment transformation as we have "hard" preprocessed the data,
            # if using npy binary files
            self.train_transformer = None
            self.val_test_transformer = None
        else:
            self.train_transformer = Compose([utils.RandomAugment(p=0.8), utils.ACPower(), utils.MFCC()])
            self.val_test_transformer = Compose([utils.RandomAugment(p=0), utils.ACPower(), utils.MFCC()])

    def train_dataloader(self):
        data = BLOND(transform=self.train_transformer, client_id=self.client_id, stage="train",
                     data_dist=self.data_dist, use_npy=self.use_npy)
        self.len_trainset = len(data)
        return DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=WORKER_COUNT)

    def val_dataloader(self):
        data = BLOND(transform=self.val_test_transformer, client_id=self.client_id, stage="val",
                     data_dist=self.data_dist, use_npy=self.use_npy)
        self.len_valset = len(data)
        return DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=WORKER_COUNT)

    def test_dataloader(self):
        data = BLOND(transform=self.val_test_transformer, client_id=self.client_id, stage="test",
                     data_dist=self.data_dist, use_npy=self.use_npy)
        self.len_testset = len(data)
        return DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=WORKER_COUNT)

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        self.batch_move_start_time = time.time()
        super(BlondLightningData, self).on_before_batch_transfer(batch, dataloader_idx)
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        super(BlondLightningData, self).on_after_batch_transfer(batch, dataloader_idx)
        try:
            self.logger.experiment.log({"timing/train/batch_move_time": time.time() - self.batch_move_start_time})
        except AttributeError:
            pass

        return batch


if __name__ == "__main__":
    dl = BlondLightningData(batch_size=128, data_dist="local", client_id=0, use_npy=True)
    dl_train = dl.train_dataloader()
    print(len(dl_train))
