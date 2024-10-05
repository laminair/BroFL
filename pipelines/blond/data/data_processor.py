import os

import numpy as np
import numpy.random
import requests
import zipfile
import tarfile
import json
import math
import torch
import h5py

import pandas as pd

from pathlib import Path
from numpy.random import dirichlet

from torchvision.transforms import Compose

try:
    from pipelines.blond import utils
except ModuleNotFoundError:
    import data_utils as utils


numpy.random.seed(42)


class BLONDPreprocessor(object):

    def __init__(self, num_clients=None, num_classes=12):
        self.file_path = os.path.dirname(os.path.realpath(__file__))

        self.indices = pd.DataFrame()
        self.trainset_indices = pd.DataFrame()
        self.valset_indices = pd.DataFrame()
        self.testset_indices = pd.DataFrame()
        self.class_map = {}
        self.num_clients = num_clients
        self.num_classes = num_classes

        self.num_clients = num_clients
        print(f"> Preparing for {self.num_clients}")

        self.folds = ["train", "val", "test"]

        Path(f"{self.file_path}/raw_data").mkdir(parents=True, exist_ok=True)
        Path(f"{self.file_path}/processed").mkdir(parents=True, exist_ok=True)

        Path(f"{self.file_path}/processed/local").mkdir(parents=True, exist_ok=True)
        Path(f"{self.file_path}/processed/iid").mkdir(parents=True, exist_ok=True)
        Path(f"{self.file_path}/processed/natural").mkdir(parents=True, exist_ok=True)
        Path(f"{self.file_path}/processed/dirichlet").mkdir(parents=True, exist_ok=True)

    def download_data(self):
        print("> Start downloading of BLOND dataset")
        if not Path(f"{self.file_path}/raw_data/federated_blond.zip").exists() and not Path(f"{self.file_path}/raw_data/events_medal.csv").exists():
            print("> The raw data is 3.6GB in size. This can take a while.")
            req = requests.get("https://syncandshare.lrz.de/dl/fi6tm2pWLRA15DB4YZrcXgrB/federated_blond.zip")
            open(f"{self.file_path}/raw_data/federated_blond.zip", "wb").write(req.content)

        print("> Downloaded BLOND data.")

    def unzip_data(self):
        if Path(f"{self.file_path}/raw_data/federated_blond.zip").exists():
            with zipfile.ZipFile(f"{self.file_path}/raw_data/federated_blond.zip", "r") as zip_f:
                Path(f"{self.file_path}/raw_data").mkdir(parents=True, exist_ok=True)
                zip_f.extractall(f"{self.file_path}/raw_data")

            zip_f.close()

            os.remove(f"{self.file_path}/raw_data/federated_blond.zip")
        print("> Done unzipping. Downloaded zip file removed.")

    def load_indices(self):
        indices_path = Path(f"{self.file_path}/raw_data/events_medal.csv")

        with open(indices_path, mode='r') as inp:
            self.indices = pd.read_csv(inp, index_col=0)
            inp.close()

        # Do some preprocessing & remove NaN values that do not carry an appliance...
        self.indices.dropna(axis=0, inplace=True)

    def create_subsets(self):
        self.indices.to_csv(f"{self.file_path}/processed/local/client_0.csv")
        print(f"> Local learning data written to disk.")

    def create_iid_splits(self):

        if len(self.folds) == 0:
            print("> IID splits have already been created.")
            return

        print(f"> Creating IID splits for {self.num_clients} clients.")

        with open(f"{self.file_path}/class_map.json", "r") as f:
            self.class_map = json.load(f)
            f.close()

        class_indices = {}
        class_len = {}

        for cl in self.class_map.keys():
            class_indices[cl] = self.indices[self.indices["Type"] == cl]
            class_len[cl] = {
                "total": len(class_indices[cl].index),
                "per_client": math.floor(len(class_indices[cl]) / self.num_clients)
            }

        for client in range(self.num_clients):
            client_indices = pd.DataFrame()

            for cl in self.class_map.keys():
                lower_bound = client * class_len[cl]["per_client"]
                upper_bound = (client + 1) * class_len[cl]["per_client"]
                subset = class_indices[cl].iloc[lower_bound: upper_bound]
                client_indices = pd.concat([client_indices, subset])

            client_indices.to_csv(f"{self.file_path}/processed/iid/client_{client}.csv")
            print(f"> IID split for client {client} written to disk.")

        print(f"> IID metadata written.")

    def create_natural_noniid_splits(self):

        if len(self.folds) == 0:
            print("> Non-IID splits have already been created.")
            return

        print(f"> Creating non-IID splits.")
        num_medals = self.indices["Medal"].max()

        with open(f"{self.file_path}/class_map.json", "r") as f:
            self.class_map = json.load(f)
            f.close()

        split_metadata = {}

        for medal in range(num_medals):

            # The medal counter starts at 1, not at 0.
            medal_indices = self.indices.loc[self.indices["Medal"] == (medal+1)]
            medal_indices.to_csv(f"{self.file_path}/processed/natural/client_{medal}.csv")

            # Register metadata
            if len(medal_indices.index) == 0:
                print(f">> No values in client {medal}")

            # Print status
            print(f"> Non-IID split for client {medal} written to disk.")

        with open(f"{self.file_path}/processed/natural/split_metadata.json", "w") as f:
            json.dump(split_metadata, f)
            f.close()

        print(f"> Non-IID metadata written.")

    def create_dirichlet_noniid_splits(self, dirichlet_alpha=0.1):
        """
        We create a dirichlet distribution based on the number of clients and the number of target classes.
        :var density: Determins the level of heterogeneity. The lower the density the more heterogeneous the data will be
        distributed.
        :return: None (stored indices on disk).
        """

        # Open most recent class map
        with open(f"{self.file_path}/class_map.json", "r") as f:
            self.class_map = json.load(f)
            f.close()

        # Open list of file indices
        indices = pd.read_csv(f"{self.file_path}/raw_data/events_medal.csv", index_col=0)

        # Creates an output of shape num_classes x num_clients
        dirichlet_sample = dirichlet(alpha=[dirichlet_alpha for _ in range(self.num_clients)], size=self.num_classes)
        dirichlet_sample.tolist()

        # Transpose class map to have numerals as keys
        class_map_transposed = {v: k for k, v in self.class_map.items()}
        clients = {}

        for client in range(self.num_clients):

            client_df = pd.DataFrame()

            for class_idx, class_dist in enumerate(dirichlet_sample):
                for fold in ["train", "val", "test"]:
                    class_name = class_map_transposed[class_idx]
                    subset = indices.loc[(indices["Type"] == class_name) & (indices["fold"] == fold)]
                    lower_bound = math.floor(len(subset) * sum(dirichlet_sample[class_idx][0:client]))
                    upper_bound = math.floor(len(subset) * sum(dirichlet_sample[class_idx][0:client + 1]))
                    subset = subset[lower_bound:upper_bound]
                    client_df = pd.concat([client_df, subset])

            client_df.to_csv(path_or_buf=f"{self.file_path}/processed/dirichlet/client_{client}.csv")
            print(f"> Dirichlet split for client {client} written to disk.")


class DataTransformator(object):
    """
    We need this class to handle the feature transformation a-priori due to a torchaudio incompatibility in PyTorch
    version 1.10.x
    """

    def __init__(self):
        self.file_path = os.path.dirname(os.path.realpath(__file__))
        self.indices = pd.read_csv(f"{self.file_path}/raw_data/events_medal.csv", index_col=0)
        self.indices = self.indices.loc[self.indices["Type"].notna()]
        self.transformer = Compose([utils.ACPower(), utils.MFCC()])

        with open(f"{self.file_path}/class_map.json", "r") as f:
            self.class_map = json.load(f)
            f.close()

    def run(self):
        """
        We iterate over all sample files, extract and preprocess the voltage and current values.
        Finally, we create a CSV file from the tensors and materialize them on disk.
        """
        for idx, row in self.indices.iterrows():
            label = row["Type"]

            if f"{label}" == "nan":
                print(idx, row)

            # We load the info right from the HDF5 file
            file_name = f"{row['Medal']}_{row['Socket']}_{row['Appliance']}_{row['Timestamp']}.h5"
            data_file = Path(f"{self.file_path}/raw_data/event_snippets/medal-{row['Medal']}/{file_name}")
            self.translate_file_to_pytorch(data_file=data_file, label=label)

            if idx % 100 == 0:
                print(f"Translated {idx} files from h5 to npy binary.")

            break

    def translate_file_to_pytorch(self, data_file, label):
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
            label = self.class_map[label]

            sample = (current, voltage, None, label)
            _, _, features, _ = self.transformer(sample)

            data_file_path = f"{data_file.absolute()}".split("/")
            medal_folder_name = data_file_path[-2]
            medal_folder = Path(f"{self.file_path}/raw_data/event_snippets_npy/{medal_folder_name}")
            medal_folder.mkdir(exist_ok=True, parents=True)

            file_name = data_file_path[-1].split(".h5")[0]

            features = features.float()
            np.save(file=f"{self.file_path}/raw_data/event_snippets_npy/{medal_folder_name}/{file_name}.npy",
                    arr=features)

        except FileNotFoundError:
            pass


def preprocess_and_sample(num_clients, dirichlet_alpha=1.0):
    """
    Since we are only working with indices, we create all data splits right away.
    All you are required to specify is the number of clients you want the data to be split onto.
    :return:
    """
    preproc = BLONDPreprocessor(num_clients=num_clients)

    # Do the preprocessing / creation of the index file
    preproc.download_data()
    preproc.unzip_data()
    preproc.load_indices()
    preproc.create_subsets()
    preproc.create_iid_splits()
    preproc.create_natural_noniid_splits()
    preproc.create_dirichlet_noniid_splits(dirichlet_alpha=dirichlet_alpha)


if __name__ == "__main__":
    preprocess_and_sample(num_clients=45, dirichlet_alpha=0.1)
    translator = DataTransformator()
    translator.run()
