import json
import os
import importlib

from pathlib import Path

import pytorch_lightning

file_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = Path(file_path).parent


def return_class_from_pythonic_string(text: str):
    """
    Translates a pythonic path into a callable module, i.e., an object.
    :param text: pythonic string, e.g. example_directory.subdirectory.file.module
    :return: callable module
    """

    path = text.split(".")

    assert len(path) > 1, "Please make sure to pass the correct format to the function: dir.subdir.file.class"

    cls = path[-1]
    pth = ""
    for idx, el in enumerate(path[:-1]):
        if idx == 0:
            pth += el
        else:
            pth += f".{el}"

    return getattr(importlib.import_module(pth), cls)


def init_device(device="client", pipeline="feblond", data_dist="iid", client_id=0, num_clients=30):
    """
    Configures data preprocess, loader, and ML model
    :param device: Specify whether we are launching a client or a server (str)
    :param pipeline: ML pipeline to evaluate (str)
    :param data_dist: Data distribution (iid/noniid) (str)
    :param client_id: Client ID to select the correct subset (int)
    :return: dict(model, dataloader, preprocessor)
    """

    if device == "server":
        # Whenever we are launching a server, we test the merged model against the global test dataset.
        data_dist = "local"
        client_id = 0

    # We load the available configurations from "pipeline_configs.json"
    with open(f"{parent_dir}/pipeline_configs.json", "r") as f:
        pipelines = json.load(f)
        f.close()

    preprocessor = return_class_from_pythonic_string(pipelines[pipeline]["preprocessor"])
    if preprocessor is not None:
        preprocessor = preprocessor(num_clients=num_clients)

    dataloader = return_class_from_pythonic_string(pipelines[pipeline]["dataloader"])(data_dist=data_dist,
                                                                                      client_id=client_id)
    ml_model = return_class_from_pythonic_string(pipelines[pipeline]["model"])()

    return {"model": ml_model, "dataloader": dataloader, "preprocessor": preprocessor}
