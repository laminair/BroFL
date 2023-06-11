import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

from datetime import datetime

import socket
import pytorch_lightning as pl
import wandb
import os
import time

from pathlib import Path
from threading import Thread

file_path = Path(f"{os.path.dirname(os.path.realpath(__file__))}")
parent_dir = file_path.parent


def get_ip_address():
    """
    Helper function that returns the true IP address.
    :return: ip address (str)
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


def init_logger(logger, wandb_entity=None, experiment_name=None, logger_suffix=None, project_group=None,
                project_name="flbench"):

    assert logger in [None, "local", "wandb"], "Please make sure to specify a valid logging option. " \
                                               "Choices: None, local, wandb"

    if logger == "wandb":
        assert wandb_entity is not None

    if logger is None:
        print("> Disabled logging.")
        return None

    project_name = f"{project_name}_{logger_suffix}"
    print(project_group)

    if logger == "wandb":
        print("> Info: Remember to save your W&B credentials to .netrc in your devices' home directories.")
        # wandb.init(entity=wandb_entity, project=project_name, name=experiment_name, group=project_group)
        logger_instance = pl.loggers.WandbLogger(entity=wandb_entity, project=project_name, # group=project_group,
                                                 name=experiment_name)
        return logger_instance
    elif logger == "local":
        return pl.loggers.TensorBoardLogger(save_dir="lightning_logs", name=experiment_name)
    else:
        return None
