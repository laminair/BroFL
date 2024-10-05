import json
import os

from pathlib import Path

file_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = Path(file_path).parent


def load_pipeline_configs():
    with open(f"{parent_dir}/pipeline_configs.json", "r") as f:
        pipelines = json.load(f)
        f.close()

    return pipelines


def get_pipeline(pipeline_name):
    pipelines = load_pipeline_configs()

    return pipelines[pipeline_name]