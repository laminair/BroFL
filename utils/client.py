import json
import os

from pathlib import Path

file_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = Path(file_path).parent


def get_client_list():

    with open(f"{parent_dir}/clients.json", "r") as f:
        clients = json.load(f)
        f.close()

    return clients


def return_client_id(device_class, ip):
    clients = get_client_list()

    return clients[device_class][ip]