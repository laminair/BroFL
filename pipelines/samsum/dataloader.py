import psutil
import torch
import os
import datasets
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from random import randrange
from datasets import concatenate_datasets

# Disable warning from the transformer library.
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


WORKER_COUNT = 12
FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class SamsumLightningData(pl.LightningDataModule):

    def __init__(self, model_name, batch_size, data_dist, client_id, source_len: int = 512,
                 target_len: int = 95, label_pad_token_id: int = -100, *args, **kwargs):
        super(SamsumLightningData, self).__init__()

        self.batch_size = batch_size
        self.data_dist = data_dist
        self.client_id = client_id

        self.source_len = source_len
        self.target_len = target_len
        self.label_pad_token_id = label_pad_token_id
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        # self.model = model

        self.trainset = self.get_dataset(client_id=self.client_id, data_dist=self.data_dist, split="train")
        self.valset = self.get_dataset(client_id=self.client_id, data_dist=self.data_dist, split="val")
        self.testset = self.get_dataset(client_id=self.client_id, data_dist=self.data_dist, split="test")

        # Data collator
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=AutoModelForSeq2SeqLM.from_pretrained(model_name),
            label_pad_token_id=self.label_pad_token_id,
            pad_to_multiple_of=8
        )

    def train_dataloader(self):
        self.len_trainset = len(self.trainset)
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=WORKER_COUNT,
                          collate_fn=self.data_collator)

    def val_dataloader(self):
        self.len_valset = len(self.valset)
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=WORKER_COUNT,
                          collate_fn=self.data_collator)

    def test_dataloader(self):
        self.len_testset = len(self.testset)
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=WORKER_COUNT,
                          collate_fn=self.data_collator)

    def get_dataset(self, client_id: int = 0, data_dist: str = "local", split: str = "train"):

        data = datasets.load_dataset(
            "json",
            data_files={"train": [f"{FILE_PATH}/data/{data_dist}/client_{client_id}.json"],
                        "val": [f"{FILE_PATH}/data/corpus/val.json"],
                        "test": [f"{FILE_PATH}/data/corpus/test.json"]
                        },
            features=datasets.Features({
                "id": datasets.Value("string"),
                "dialogue": datasets.Value("string"),
                "summary": datasets.Value("string"),
            })
        )

        tokenized_data = data.map(
            self.preprocess,
            batched=True,
            remove_columns=["dialogue", "summary", "id"]
        )

        return tokenized_data[split]

    def preprocess(self, item, padding: str = "max_length"):
        # add prefix to the input for t5
        inputs = ["summarize: " + item for item in item["dialogue"]]

        # tokenize inputs. The "max length" argument is pre-computed from the exec part below.
        model_inputs = self.tokenizer(inputs, max_length=self.source_len, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = self.tokenizer(text_target=item["summary"], max_length=self.target_len, padding=padding,
                                truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


if __name__ == "__main__":

    # Load dataset from disk. We need this as we split the dataset into client subsets.
    dataset = datasets.load_dataset(
        "json",
        data_files={
            "train": [f"{FILE_PATH}/data/corpus/train.json"],
            "val": [f"{FILE_PATH}/data/corpus/val.json"],
            "test": [f"{FILE_PATH}/data/corpus/test.json"]
        },
        features=datasets.Features({
            "id": datasets.Value("string"),
            "dialogue": datasets.Value("string"),
            "summary": datasets.Value("string"),
        })
    )

    # Extract a sample to get familiar with the data shape.
    sample = dataset['train'][randrange(len(dataset["train"]))]
    print(f"dialogue: \n{sample['dialogue']}\n---------------")
    print(f"summary: \n{sample['summary']}\n---------------")

    # Get FLAN T5 model and tokenizer from Huggingface.
    model_id = "google/flan-t5-base"
    # Load tokenizer of FLAN-t5-base
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"]
    )
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")

    # Test dataloader
    dataloader = SamsumLightningData(
        model=AutoModelForSeq2SeqLM.from_pretrained(model_id),
        tokenizer=tokenizer,
        batch_size=2,
        data_dist="dirichlet",
        client_id=0,
        model_name=model_id,
        source_len=512, target_len=95
    )

    print(f"Batch size validation: {len(list(iter(dataloader.train_dataloader())))}")
