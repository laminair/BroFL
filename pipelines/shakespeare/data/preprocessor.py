"""
Data was preprocessed the exact same ways as provided in the LEAF benchmark.
See here for details: https://github.com/TalwalkarLab/leaf/tree/master/data/shakespeare
"""

import json
import os
import math
from numpy.random import dirichlet
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


class ShakespeareSampler(object):

    def __init__(self, num_clients=48, dirichlet_density=1.0):
        self.file_path = os.path.dirname(os.path.realpath(__file__))

        self.data_dists = ["local", "iid", "dirichlet"]
        self.num_clients = num_clients
        self.folds = ["train", "val", "test"]
        self.dirichlet_density = dirichlet_density

        Path(f"{self.file_path}/raw_data").mkdir(parents=True, exist_ok=True)
        Path(f"{self.file_path}/processed").mkdir(parents=True, exist_ok=True)

        Path(f"{self.file_path}/processed/local").mkdir(parents=True, exist_ok=True)
        Path(f"{self.file_path}/processed/iid").mkdir(parents=True, exist_ok=True)
        Path(f"{self.file_path}/processed/natural").mkdir(parents=True, exist_ok=True)
        Path(f"{self.file_path}/processed/dirichlet").mkdir(parents=True, exist_ok=True)

    def sample(self):
        self.load_dataset_from_json()
        self.get_labels()
        self.assign_folds()
        # self.generate_subset_for_baseline_testing(num_samples=10000)
        self.create_iid_splits()
        self.create_dirichlet_noniid_splits()

    def load_dataset_from_json(self):

        if Path(f"{self.file_path}/processed/client_0.csv").exists():
            # CSV set already loaded.
            return

        with open(f"{self.file_path}/raw_data/all_data.json", "r") as f:
            raw_data = json.load(f)
            f.close()

        users = raw_data["users"]
        user_data = raw_data["user_data"]

        self.processed_data = pd.DataFrame()
        for user in users:
            user_text = user_data[user]["x"]
            user_label = user_data[user]["y"]
            user_df = pd.DataFrame({"user": user, "text": user_text, "next_char": user_label})

            self.processed_data = pd.concat([self.processed_data, user_df])

        raw_data = None  # Throw raw data out of memory.
        self.processed_data.to_csv(path_or_buf=f"{self.file_path}/processed/local/client_0.csv")

    def get_labels(self):

        self.processed_data = pd.read_csv(filepath_or_buffer=f"{self.file_path}/processed/local/client_0.csv", index_col=0)
        self.labels = self.processed_data["next_char"].unique().tolist()

        char_dict = {}
        for idx, char in enumerate(self.labels):
            char_dict[char] = idx

        with open(f"{self.file_path}/raw_data/vocabulary.json", "w") as f:
            json.dump(char_dict, f, indent=2)
            f.close()

    def assign_folds(self):
        folded_set = pd.DataFrame()
        trainset, testset = train_test_split(self.processed_data, train_size=0.8, test_size=0.2)
        valset, testset = train_test_split(testset, train_size=0.5, test_size=0.5)
        trainset["fold"] = "train"
        valset["fold"] = "val"
        testset["fold"] = "test"
        folded_set = pd.concat([folded_set, trainset, valset, testset])

        self.processed_data = folded_set
        folded_set.to_csv(path_or_buf=f"{self.file_path}/processed/local/client_0.csv")
        folded_set = None

    def generate_subset_for_baseline_testing(self, num_samples=10_000):
        self.processed_data = pd.read_csv(f"{self.file_path}/processed/local/client_0.csv")

        max_train_idx = int(0.8 * num_samples)
        max_val_idx = int(0.1 * num_samples)
        max_test_idx = int(0.1 * num_samples)

        trainset = self.processed_data.loc[self.processed_data["fold"] == "train"].iloc[0: max_train_idx]
        valset = self.processed_data.loc[self.processed_data["fold"] == "val"].iloc[0: max_val_idx]
        testset = self.processed_data.loc[self.processed_data["fold"] == "test"].iloc[0: max_test_idx]
        subset = pd.concat([trainset, valset, testset])
        subset.to_csv(path_or_buf=f"{self.file_path}/processed/local/client_0.csv")
        subset = None

    def create_iid_splits(self):

        print(f"> Creating IID splits for {self.num_clients} clients.")
        self.processed_data = pd.read_csv(filepath_or_buffer=f"{self.file_path}/processed/local/client_0.csv",
                                          index_col=0)

        class_indices = {}
        class_len = {}

        for label in self.labels:
            class_indices[label] = {}
            class_len[label] = {}
            for fold in self.folds:
                class_indices[label][fold] = self.processed_data[(self.processed_data["next_char"] == label) & (self.processed_data["fold"] == fold)]
                class_len[label][fold] = {
                    "total": len(class_indices[label][fold]),
                    "per_client": math.floor(len(class_indices[label][fold]) / self.num_clients)
            }

        for client in range(self.num_clients):
            client_indices = pd.DataFrame()

            for fold in self.folds:
                for label in self.labels:
                    lower_bound = client * class_len[label][fold]["per_client"]
                    upper_bound = (client + 1) * class_len[label][fold]["per_client"]
                    subset = class_indices[label][fold].iloc[lower_bound: upper_bound]
                    client_indices = pd.concat([client_indices, subset])

            client_indices.to_csv(f"{self.file_path}/processed/iid/client_{client}.csv")
            print(f"> IID split for client {client} written to disk. {len(client_indices)} data points.")

    def create_dirichlet_noniid_splits(self):
        # Creates an output of shape num_classes x num_clients
        dirichlet_sample = dirichlet([self.dirichlet_density for _ in range(self.num_clients)], size=len(self.labels))
        dirichlet_sample = dirichlet_sample.tolist()
        self.processed_data = pd.read_csv(f"{self.file_path}/processed/local/client_0.csv", index_col=0)

        for client in range(self.num_clients):

            client_df = pd.DataFrame()

            for class_idx, class_dist in enumerate(dirichlet_sample):
                for fold in self.folds:
                    subset = self.processed_data.loc[(self.processed_data["next_char"] == self.labels[class_idx]) & (self.processed_data["fold"] == fold)]
                    lower_bound = math.floor(len(subset) * sum(dirichlet_sample[class_idx][0:client]))
                    upper_bound = math.floor(len(subset) * sum(dirichlet_sample[class_idx][0:client + 1]))
                    subset = subset[lower_bound:upper_bound]
                    client_df = pd.concat([client_df, subset])

            client_df.to_csv(path_or_buf=f"{self.file_path}/processed/dirichlet/client_{client}.csv")
            print(f"> Dirichlet split for client {client} written to disk. {len(client_df)} data points.")

        print(">>> Done writing dirichlet splits.")


if __name__ == "__main__":
    sampler = ShakespeareSampler(num_clients=45, dirichlet_density=0.1)
    sampler.sample()
