"""
This script create a non-IID split of the samsum dataset.
We only consider non-IIDness as the number of samples on a client as we do not have labels in a classical sense.
"""
import json
import math
import os

import pandas as pd
from numpy.random import dirichlet


class SamsumPreprocessor(object):
    """
    Dataset sampler for non-IID federated learning experiments.
    The corpus dataset can be found here: https://huggingface.co/datasets/samsum/blob/main/data/corpus.7z
    """
    def __init__(self, dirichlet_alpha: float = 1.0, num_clients: int = 10):
        self.file_path = os.path.dirname(os.path.realpath(__file__))
        self.alpha = dirichlet_alpha
        self.num_clients = num_clients

        self.trainset: list = []
        self.valset: list = []
        self.testset: list = []

    def load_datasets(self):
        with open(f"{self.file_path}/corpus/train.json", "r") as f:
            self.trainset = json.load(fp=f)
            f.close()

    def generate_local_dataset(self):
        print(f"Writing local training dataset...")
        with open(f"{self.file_path}/local/client_0.json", "w") as f:
            json.dump(self.trainset, fp=f, indent=1)
            f.close()

    def generate_dirichlet_split(self):
        # Creates an output of shape 1 x num_clients
        # Since we do not have a classification task at hand, we split the dataset by the number of samples only.

        dirichlet_sample = dirichlet(alpha=[self.alpha for _ in range(self.num_clients)], size=1)
        dirichlet_sample.tolist()

        len_trainset = len(self.trainset)
        lower_bound = 0
        for client_id, sample_rate in enumerate(dirichlet_sample[0]):
            upper_bound = lower_bound + math.floor(len_trainset * sample_rate)

            subset: list = self.trainset[lower_bound: upper_bound]
            lower_bound = upper_bound

            print(f"Writing subset for client {client_id}")
            with open(f"{self.file_path}/dirichlet/client_{client_id}.json", "w") as f:
                json.dump(subset, fp=f, indent=1)
                f.close()


if __name__ == "__main__":
    preprocessor = SamsumPreprocessor(dirichlet_alpha=1.0, num_clients=10)
    preprocessor.load_datasets()
    preprocessor.generate_dirichlet_split()

