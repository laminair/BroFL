import os
import pytorch_lightning as pl
import pandas as pd
import sys
import torch
import psutil
import time

from torch.utils.data import Dataset, DataLoader
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))


WORKER_COUNT = 2


class ShakespeareDataset(Dataset):
    def __init__(self, client_id, data_dist, fold, *args, **kwargs):
        """get `Dataset` for shakespeare dataset
        Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): sentence list data
            targets (list): next-character target list
        """
        self.file_path = self.file_path = os.path.dirname(os.path.realpath(__file__))
        self.data_path = Path(f"{self.file_path}/data/processed/{data_dist}")
        self.data_path = Path(f"{self.data_path}/client_{0 if data_dist == 'local' else client_id}.csv")
        # self.vocab_path = Path(f"{self.file_path}/data/raw_data/vocabulary.json")

        self.client_id = 0 if data_dist == "local" else client_id

        self.data = pd.read_csv(f"{self.data_path}")
        self.data = self.data.loc[self.data["fold"] == fold]
        self.data = self.data.sample(frac=1).reset_index()

        # We use the vocab suggested in
        # https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
        voc = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        self.vocab = {char: idx for idx, char in enumerate([*voc])}
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sentence = self.data.iloc[item]["text"]
        sentence = torch.tensor(self._sentence_to_indices(sentence, self.vocab))
        label = self.data.iloc[item]["next_char"]
        label = torch.tensor(self._label_to_index(label, self.vocab))

        return sentence, label

    @staticmethod
    def _sentence_to_indices(sentence, vocab):
        indices = []
        for c in sentence:
            indices.append(vocab.get(c))
        return indices

    @staticmethod
    def _label_to_index(label, vocab):
        return vocab.get(label)


class ShakespeareLightningData(pl.LightningDataModule):

    def __init__(self, batch_size, data_dist="local", client_id=0, *args, **kwargs):
        super(ShakespeareLightningData, self).__init__()
        self.batch_size = batch_size
        self.data_dist = data_dist
        self.client_id = client_id

    def train_dataloader(self):
        data = ShakespeareDataset(client_id=self.client_id, data_dist=self.data_dist, fold="train")
        self.len_trainset = len(data)
        return DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=WORKER_COUNT)

    def val_dataloader(self):
        data = ShakespeareDataset(client_id=self.client_id, data_dist=self.data_dist, fold="val")
        self.len_valset = len(data)
        return DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=WORKER_COUNT)

    def test_dataloader(self):
        data = ShakespeareDataset(client_id=self.client_id, data_dist=self.data_dist, fold="test")
        self.len_testset = len(data)
        return DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=WORKER_COUNT)

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        self.batch_move_start_time = time.time()
        super(ShakespeareLightningData, self).on_before_batch_transfer(batch, dataloader_idx)
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        super(ShakespeareLightningData, self).on_after_batch_transfer(batch, dataloader_idx)
        try:
            self.logger.experiment.log({"timing/train/batch_move_time": time.time() - self.batch_move_start_time})
        except AttributeError:
            pass

        return batch


if __name__ == "__main__":
    dataset = ShakespeareLightningData(client_id=0, data_dist="dirichlet", batch_size=128)
    print(next(iter(dataset.train_dataloader())))