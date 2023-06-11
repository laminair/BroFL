import pandas as pd
import requests
import zipfile
import math
import json
import os

from pathlib import Path
from torchvision.datasets.mnist import read_label_file, read_image_file
from torchvision.datasets.utils import extract_archive
from torchvision import transforms as T
from collections import OrderedDict
from numpy.random import dirichlet

from typing import Dict


class MNISTPreprocessor():

    def __init__(self):
        self.download_link = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
        self.target_split = "byclass"
        self.mapping_file = "mapping.txt"
        self.folds = ["train", "test"]
        self.file_names = ["images-idx3-ubyte", "labels-idx1-ubyte"]
        self.img_transform = T.ToPILImage()
        self.lbl_dict = OrderedDict()

        self.file_path = os.path.dirname(os.path.realpath(__file__))
        self.raw_data_path = Path(f"{self.file_path}/raw_data")
        self.local_data_path = Path(f"{self.file_path}/processed/local")
        self.iid_data_path = Path(f"{self.file_path}/processed/iid")
        self.natural_path = Path(f"{self.file_path}/processed/natural")
        self.dirichlet_path = Path(f"{self.file_path}/processed/dirichlet")

        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.local_data_path.mkdir(parents=True, exist_ok=True)
        self.iid_data_path.mkdir(parents=True, exist_ok=True)
        self.natural_path.mkdir(parents=True, exist_ok=True)
        self.dirichlet_path.mkdir(parents=True, exist_ok=True)

        Path(f"{self.file_path}/raw_data/images/train").mkdir(parents=True, exist_ok=True)
        Path(f"{self.file_path}/raw_data/images/val").mkdir(parents=True, exist_ok=True)
        Path(f"{self.file_path}/raw_data/images/test").mkdir(parents=True, exist_ok=True)

        self.dataset_path = Path(f"{self.file_path}/raw_data/emnist.zip")
        self.dataset_extracted_path = Path(f"{self.file_path}/raw_data/gzip")
        self.dataset_extracted_images_path = Path(f"{self.file_path}/raw_data/images")

    def download_dataset(self):
        if not self.dataset_path.exists() and not self.dataset_extracted_path.exists():
            print("> Start downloading of FEMNIST dataset")
            req = requests.get("http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip")
            open(f"{self.file_path}/raw_data/emnist.zip", "wb").write(req.content)

            print("> Downloaded FEMNIST data by class and writer")

    def unzip_dataset(self):
        if Path(f"{self.file_path}/raw_data/emnist.zip").exists():
            print("> Start unzipping")
            with zipfile.ZipFile(f"{self.file_path}/raw_data/emnist.zip", "r") as zip_f:
                Path(f"{self.file_path}/raw_data").mkdir(parents=True, exist_ok=True)
                zip_f.extractall(f"{self.file_path}/raw_data")
                zip_f.close()

            os.remove(f"{self.file_path}/raw_data/emnist.zip")
            print("> Done unzipping")

    def untar_target_split(self):
        if self.dataset_extracted_path.exists() and self.dataset_extracted_images_path.exists():
            return

        print(f"> Start untaring for split {self.target_split}...")
        for fold in self.folds:
            for file in self.file_names:
                from_path = f"{self.file_path}/raw_data/gzip/emnist-{self.target_split}-{fold}-{file}"

                if not Path(from_path).exists():
                    try:
                        extract_archive(from_path=f"{from_path}.gz")
                        os.remove(f"{from_path}.gz")
                    except FileNotFoundError:
                        pass
                else:
                    print(f"> {from_path} has already been untared.")

        print(f"> Done untaring split {self.target_split}.")

    def load_target_split_and_extract(self):
        indices_pd = pd.DataFrame()
        for fold in self.folds:
            lbl_file = read_label_file(f"{self.file_path}/raw_data/gzip/emnist-{self.target_split}-{fold}-labels-idx1-ubyte")
            img_file = read_image_file(f"{self.file_path}/raw_data/gzip/emnist-{self.target_split}-{fold}-images-idx3-ubyte")

            n_images = len(img_file)
            n_labels = len(lbl_file)

            assert n_images == n_labels, "Images and Labels must be of equal lenght."

            idx = 0
            for label in lbl_file:
                label = int(label)
                if label not in self.lbl_dict.keys():
                    self.lbl_dict[label] = idx
                    idx += 1

            indices = []
            for idx in range(n_images):
                row = self.write_image_to_file(fold, label_file=lbl_file, image_file=img_file, index=idx)
                indices.append(row)

            indices = pd.DataFrame(indices)

            indices_pd = pd.concat([indices_pd, indices])

        indices_pd.to_csv(f"{self.file_path}/processed/local/client_0.csv")
        print("> Done writing images to files.")

    def write_image_to_file(self, fold, label_file, image_file, index):
        img = image_file[index]
        lab = label_file[index]

        if fold == "train" and index % 10 == 9:
            fold = "val"

        img_name = f"{fold}-{lab}-{index}.png"
        img_path = f"raw_data/images/{fold}/{img_name}"

        if not Path(f"{self.file_path}/{img_path}").exists():
            pil_imge = self.img_transform(img)
            pil_imge.save(f"{self.file_path}/{img_path}", format="PNG")

        return {"file_name": img_name, "path": img_path, "class": int(lab), "class_id": self.lbl_dict[int(lab)], "fold": fold}


class MNISTSampler(object):

    def __init__(self, num_clients=48, dirichlet_density=1.0):
        self.file_path = os.path.dirname(os.path.realpath(__file__))
        self.indices = pd.read_csv(filepath_or_buffer=f"{self.file_path}/processed/local/client_0.csv")

        self.num_clients = num_clients
        self.density = dirichlet_density
        self.folds = ["train", "val", "test"]

    def create_iid_splits(self):
        class_names = self.indices["class"].unique().tolist()

        for client in range(self.num_clients):
            client_indices = pd.DataFrame()
            for class_name in class_names:
                for fold in self.folds:
                    subset = self.indices.loc[(self.indices["class"] == class_name) & (self.indices["fold"] == fold)]
                    items_per_client = math.floor(len(subset) / self.num_clients)
                    lower_bound = client * items_per_client
                    upper_bound = (client + 1) * items_per_client

                    subset = subset.iloc[lower_bound:upper_bound, :]
                    client_indices = pd.concat([client_indices, subset])

            client_indices = client_indices.sample(frac=1).reset_index(drop=True)
            client_indices.to_csv(path_or_buf=f"{self.file_path}/processed/iid/client_{client}.csv")

        print(f"> IID splits for {self.num_clients} written to disk.")

    def create_natural_noniid_splits(self):
        class_names = self.indices["class"].unique().tolist()

        for client, class_name in enumerate(class_names):
            client_indices = self.indices.loc[(self.indices["class"] == class_name)]
            client_indices = client_indices.sample(frac=1).reset_index(drop=True)
            client_indices.to_csv(path_or_buf=f"{self.file_path}/processed/natural/client_{client}.csv")

        print(f"> Natural non-IID splits for {len(class_names)} clients written to disk.")

    def create_dirichlet_noniid_splits(self, density=1):
        """
        We create a dirichlet distribution based on the number of clients and the number of target classes.
        :var density: Determines the level of heterogeneity. The lower the density the more heterogeneous the data will be
        distributed.
        :return: None (stored indices on disk).
        """
        class_ids = self.indices["class_id"].unique().tolist()

        # Creates an output of shape num_classes x num_clients
        dirichlet_sample = dirichlet(alpha=[self.density for _ in range(self.num_clients)], size=len(class_ids))
        dirichlet_sample.tolist()

        for client in range(self.num_clients):
            client_indices = pd.DataFrame()
            for class_id, class_dist in enumerate(dirichlet_sample):
                for fold in self.folds:
                    subset = self.indices.loc[(self.indices["class_id"] == class_id) & (self.indices["fold"] == fold)]
                    lower_bound = math.floor(len(subset) * sum(dirichlet_sample[class_id][0:client]))
                    upper_bound = math.floor(len(subset) * sum(dirichlet_sample[class_id][0:client + 1]))

                    subset = subset.iloc[lower_bound:upper_bound, :]
                    client_indices = pd.concat([client_indices, subset])

            client_indices = client_indices.sample(frac=1).reset_index(drop=True)
            client_indices.to_csv(path_or_buf=f"{self.file_path}/processed/dirichlet/client_{client}.csv")

        print(f"> Dirichlet non-IID splits for {self.num_clients} written to disk.")


def preprocess_and_sample(num_clients=45, dirichlet_density=1.0):

    proc = MNISTPreprocessor()
    proc.download_dataset()
    proc.unzip_dataset()
    proc.untar_target_split()
    proc.load_target_split_and_extract()

    samp = MNISTSampler(num_clients=45, dirichlet_density=dirichlet_density)
    samp.create_iid_splits()
    samp.create_natural_noniid_splits()
    samp.create_dirichlet_noniid_splits()


if __name__ == "__main__":
    preprocess_and_sample(num_clients=45, dirichlet_density=0.1)
