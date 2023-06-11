"""
This script automates all FLBench experiments and ensures reproducibility.
"""
import argparse
import datetime
import json
import re
import socket

import paramiko
import time
import logging
import os
import random

from threading import Thread
from configparser import ConfigParser
from pathlib import Path

from ansible.parsing.dataloader import DataLoader
from ansible.inventory.manager import InventoryManager


class ExperimentAutomatorBase(object):

    def __init__(self, inventory_path=None, server_endpoint=None, experiment_plan_path=None, dryrun=False):
        self.file_path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.remote_flbench_directory = "/opt/flbench"
        self.inventory_path = inventory_path

        # Read Pipeline Configurations
        self.pipeline_configs, \
        self.fed_optimizer_configs, \
        self.auxiliary_config, \
        self.automation_config, \
        self.privacy_config = self.read_global_config()

        self.device_ip_addresses = ConfigParser()
        self.threads_dict = {}

        self.inventory = self.load_inventory(inventory_path=self.inventory_path)
        self.experiment_plan = experiment_plan_path
        self.server_endpoint = server_endpoint

        self.dryrun = dryrun

        # Auxiliary configuration config
        self.logger = self.auxiliary_config["logger"]
        self.logging_frequency = self.auxiliary_config["logging_freq"]

        # Experiment automation config
        self.num_devices = self.automation_config["num_devices"]
        self.client_dropout_rates = self.automation_config["client_dropouts_per_round"]
        self.experiment_repetitions = self.automation_config["experiment_repetitions"]
        self.available_ports = self.automation_config["server_ports"]
        self.tmux_session_name = self.automation_config["tmux_session_name"]
        self.tmux_health_check_freq = self.automation_config["tmux_check_alive_experiment_automation_minutes"]
        self.include_dp_experiments = self.automation_config["use_differential_privacy"]
        self.inventory_source = self.automation_config["inventory_source"]
        self.training_rounds = self.automation_config["fl_training_rounds"]

        # Here we configure whether we want to use differential privacy or not.
        if self.include_dp_experiments:
            self.dp_experiments = [False, True]
        else:
            self.dp_experiments = [False]

    def launch(self):

        self.log_information(f"Start time: {datetime.datetime.now()}.")

        # Get list of all client ip addresses for experiments. Note: IP addresses will be used in order as they appear
        # in the list.
        clients = []
        for group, ip_addr_list in self.inventory.items():
            if "flbench_client" in group:
                clients += ip_addr_list

        if self.dryrun:
            # When dry running we only generate the experiment plan.
            experiment_plan = self.generate_experiment_plan(clients=clients)
            print(f"Generated experiment plan with {len(experiment_plan)} experiments. Dry run: {self.dryrun}")
            with open(f"{self.file_path}/experiment-plan.json", "w") as f:
                json.dump(experiment_plan, fp=f, indent=2)
                f.close()

            return

        if not Path(f"{self.file_path}/experiment-plan.json").exists() and self.experiment_plan is None:
            raise FileNotFoundError("Please generate an experiment plan first, place 'experiment_plan.json' in the "
                                    "automation/experiments directory. If you wish to use a custom experiment plan pass"
                                    " the path to the json file with the --experiment-plan-path flag.")
        elif self.experiment_plan is None:
            with open(f"{self.file_path}/experiment-plan.json", "r") as f:
                self.experiment_plan = json.load(f)
                f.close()
        else:
            with open(f"{self.experiment_plan}", "r") as f:
                self.experiment_plan = json.load(f)
                f.close()

        # Update the GIT repository on remotes to have the most recent commit.
        server_ip = self.server_endpoint.split(":")[0]
        self.update_git_repo_multithread(ips=clients, server=server_ip)
        time.sleep(30)

        experiment_counter = 1
        for repetition in range(self.experiment_repetitions):
            for idx, experiment in enumerate(self.experiment_plan):

                experiment_name = self.generate_experiment_name(
                    pipeline=experiment["pipeline"],
                    ml_model=experiment["model"],
                    data_dist=experiment["data_dist"],
                    num_clients=experiment["devices"],
                    optimizer=experiment["optimizer"],
                    inventory_name=self.inventory_path,
                    dropout_rate=experiment["dropout_rate"],
                    training_rounds=experiment["training_rounds"],
                    use_dp=experiment["use_dp"],
                    noise_multiplier=experiment["noise_multiplier"]
                )

                self.log_information(
                    f"Repetition {repetition + 1}. "
                    f"Pipeline {experiment['pipeline']}. "
                    f"Model {experiment['model']}. "
                    f"Optimizer {experiment['optimizer']}. "
                    f"DataDist {experiment['data_dist']}. "
                    f"DeviceNum {experiment['devices']}. "
                    f"DropoutNb {experiment['dropout_rate']}. "
                    f"ExperimentCounter {experiment_counter}."
                )

                if len(clients) == 1:
                    # When training on a single machine with a multitude of clients, we need to provide a client IP
                    # list for the client spawn process to have enough enpoints available.
                    clients = [clients[0] for _ in range(experiment["devices"])]

                for client_id, client_ip in enumerate(clients[:experiment['devices']]):
                    # We still need to pass arguments to the client as training local baselines
                    # launches a client without a server. In FL setups the server would transmit
                    # the correct client configuration.
                    self.configure_and_launch_client(
                        client_id=client_id,
                        client_ip=client_ip,
                        data_dist=experiment["data_dist"],
                        pipeline=experiment["pipeline"],
                        ml_model=experiment["model"],
                        experiment_name=experiment_name
                    )

                    # When running all clients on a single machine you need to give time for the SSH sessions to
                    # build up
                    time.sleep(1)

                time.sleep(30)  # Wait 30sec until the next experiment
                self.join_threads()
                experiment_counter += 1

    @staticmethod
    def eval_skip_conditions(pipeline, model, optimizer, data_dist, devices, dropout_rate, clients,
                             fl_optimizer_index, use_dp, noise_multiplier):

        if pipeline == "tinyimagenet200" or pipeline == "cinic10" or pipeline == "imdbreviews" or pipeline == "sentiment140":
            # We only experiment with MNIST (CV), BLOND (NILM), and Shakespeare (NLP) for the FLBench paper.
            return True

        if optimizer == "fedpaq":
            # We exclude FedProx and FedPAQ for now. FedPAQ is a client-side quantizer.
            return True

        if dropout_rate > 0 and (data_dist == "iid" or data_dist == "local"):
            # DP does not make sense in a local training setup. Also, there are no dropouts in local baseline exp's.
            # We cannot run DP experiment in dropout settings as DP currently requires all clients to remain available
            # in a training round.
            # We do not perform a dropout analysis for IID experiments as all clients have the same data distribution.
            return True

        if noise_multiplier > 0 and use_dp is False:
            return True

        if noise_multiplier == 0:
            return True

        if use_dp is True and (data_dist != "dirichlet" or optimizer == "fedavg" or noise_multiplier == 0 or dropout_rate > 0):
            return True

    def generate_experiment_plan(self, clients):
        experiment_plan = []
        for f, use_dp in enumerate(self.dp_experiments):
            for g, noise in enumerate(self.privacy_config["noise_multiplier"]):
                for h, training_rnds in enumerate(self.training_rounds):
                    # self.exp_plan_has_local = False
                    for i, (pipeline, pipe_cfg) in enumerate(self.pipeline_configs.items()):
                        for j, (model, _) in enumerate(pipe_cfg["models"].items()):
                            for k, (optimizer, optim_cfg) in enumerate(self.fed_optimizer_configs.items()):
                                for l, data_dist in enumerate(pipe_cfg["dataloader"]["available_data_dists"]):
                                    for m, devices in enumerate(self.num_devices):
                                        for n, dropout_rate in enumerate(self.client_dropout_rates):

                                            skip_condition = self.eval_skip_conditions(optimizer=optimizer,
                                                                                       devices=devices,
                                                                                       data_dist=data_dist,
                                                                                       clients=clients,
                                                                                       dropout_rate=dropout_rate,
                                                                                       model=model,
                                                                                       use_dp=use_dp,
                                                                                       noise_multiplier=noise,
                                                                                       fl_optimizer_index=k,
                                                                                       pipeline=pipeline)

                                            if skip_condition is True:
                                                continue

                                            print(f"Experiment: DP - {use_dp}. Noise - {noise}. Rounds - {training_rnds}. Pipe - {pipeline}. Model - {model}. Opt - {optimizer}. Dist - {data_dist}. Devs - {devices}. Rate - {dropout_rate}.")

                                            experiment = {
                                                "use_dp": use_dp,
                                                "noise_multiplier": noise,
                                                "training_rounds": 1 if data_dist == "local" else training_rnds,
                                                "pipeline": pipeline,
                                                "model": model,
                                                "optimizer": None if data_dist == "local" else optimizer,
                                                "data_dist": data_dist,
                                                "devices": 1 if data_dist == "local" else devices,
                                                "dropout_rate": dropout_rate
                                            }
                                            experiment_plan.append(experiment)

        return experiment_plan

    def configure_and_launch_client(self, client_ip, experiment_name, client_id=None, data_dist=None, pipeline=None,
                                    ml_model=None):

        kwargs = {
            "client_id": client_id,
            "client_ip": client_ip,
            "data_dist": data_dist,
            "pipeline": pipeline,
            "ml_model": ml_model,
            "experiment_name": experiment_name
        }
        target_fn = self.launch_client
        self.create_thread(name=f"{experiment_name}_{client_ip}_{client_id}", target=target_fn, kwargs=kwargs)

    def launch_client(self, client_ip, experiment_name, client_id=None, pipeline=None, ml_model=None, data_dist=None):

        if "raspi" in client_ip or "inventory.static" in experiment_name:
            # We need this to avoid a Memory allocation error on AARCH64 devices.
            # See: https://github.com/opencv/opencv/issues/14884 & https://stackoverflow.com/q/67735216
            command_addon = f"LD_PRELOAD='/opt/flbench/venv/lib/python3.9/site-packages/scikit_learn.libs/" \
                            f"libgomp-d22c30c5.so.1.0.0'"
        else:
            command_addon = ""

        # We run our venv python with sudo rights to allow scapy network traffic monitoring.
        command = f"{command_addon} {self.remote_flbench_directory}/venv/bin/python3 " \
                  f"{self.remote_flbench_directory}/device.py "

        if data_dist == "local":
            # We only send a configuration to clients if they train the local baseline.
            command += f"--pipeline {pipeline} " \
                       f"--ml-model {ml_model} " \
                       f"--data-dist {data_dist} " \
                       f"--client-id {client_id} " \
                       f"--experiment-name {experiment_name} " \
                       f"\n"  # Do not remove this line!
            print(f"Client command: {command}")
        else:
            # In federated settings, clients are treated ephemeral. If you pass additional arguments, they will be
            # ignored in the current implementation.
            command += f"--server-endpoint {self.server_endpoint}" \
                       f"\n"  # Do not remove this line!

        tmux_session_name = self.automation_config["tmux_session_name"]
        command = f"tmux new-session -d -s '{tmux_session_name}-{client_id}' '{command}'"
        status = self.send_client_command(node=client_ip, cmd=command)
        time.sleep(5)  # Wait 5sec for the tmux session to come alive
        self.check_if_tmux_session_is_alive(node=client_ip, target_session=f"{tmux_session_name}-{client_id}")
        return

    def configure_and_launch_server(self, pipeline, ml_model, data_dist, use_dp, noise_multiplier, num_devices,
                                    training_rounds, client_dropout_rate, optimizer, experiment_name):
        kwargs = {
            "pipeline": pipeline,
            "ml_model": ml_model,
            "n_clients": num_devices,
            "fl_strategy": optimizer,
            "experiment_name": experiment_name,
            "data_dist": data_dist,
            "use_dp": use_dp,
            "dp_noise_multiplier": noise_multiplier,
            "training_rounds": training_rounds,
            "client_dropout_rate": client_dropout_rate,
        }

        target_fn = self.launch_server
        self.create_thread(name=experiment_name, target=target_fn, kwargs=kwargs)
        time.sleep(30)  # Wait 30sec for the server to come alive

    def launch_server(self, pipeline, ml_model, data_dist, use_dp, n_clients, training_rounds, client_dropout_rate,
                      fl_strategy, dp_noise_multiplier, experiment_name):
        # server_port = self.set_fl_server_port()
        server_ip = self.server_endpoint.split(":")[0]
        command = f"{self.remote_flbench_directory}/venv/bin/python3 {self.remote_flbench_directory}/server.py " \
                  f"--pipeline {pipeline} " \
                  f"--ml-model {ml_model} " \
                  f"--data-dist {data_dist} " \
                  f"--dp-noise-multiplier {dp_noise_multiplier} " \
                  f"--server-endpoint {self.server_endpoint} " \
                  f"--num-clients {n_clients} " \
                  f"--training-rounds {training_rounds} " \
                  f"--client-dropouts {client_dropout_rate} " \
                  f"--fl-strategy {fl_strategy} " \
                  f"--experiment-name {experiment_name} "

        if use_dp == True:
            command += f"--use-dp {use_dp} "

        command += f"\n"  # Do not remove this line!

        print(f"Server command: {command}")
        tmux_session_name = self.automation_config["tmux_session_name"]
        command = f"tmux new-session -d -s '{tmux_session_name}' '{command}'"
        status = self.send_client_command(node=server_ip, cmd=command)
        time.sleep(5)  # Wait 5sec for the tmux session to come alive
        self.check_if_tmux_session_is_alive(node=server_ip, target_session=tmux_session_name)

        return

    def load_inventory(self, inventory_path):
        """
        This method loads the ansible inventory/inventories located in the ansible automation folder.
        If no inventory path is provided, we use the default inventory.cfg and inventory_static.cfg.
        :var inventory_path: str. Absolute path to a custom inventory. Default: None.
        :return: dict[(host_group, ip_addresses)]
        """
        if inventory_path is not None:
            inventory_paths = [inventory_path]
        else:
            dynamic_inventory = f"{self.file_path.parent.parent}/automation/ansible/inventory.cfg"
            static_inventory = f"{self.file_path}/automation/ansible/inventory_static.cfg"
            inventory_paths = [dynamic_inventory, static_inventory]

        data_loader = DataLoader()
        # This returns all device addresses grouped by their purpose (orchestrator, server, vm)
        inventory = InventoryManager(loader=data_loader, sources=inventory_paths)
        return inventory.get_groups_dict()

    @staticmethod
    def log_information(msg) -> None:
        logging.info(f"{msg}")
        print(f"{msg}")

    def read_global_config(self):
        with open(f"{self.file_path.parent.parent}/config.json", "r") as f:
            global_config = json.load(f)
            f.close()

        pipeline_configs = global_config["pipelines"]
        fed_optimizer_configs = global_config["federated_optimizer"]
        auxiliary_config = global_config["auxiliary"]
        automation_config = global_config["experiment_automation"]
        privacy_config = global_config["privacy_techniques"]["differential_privacy"]["user_level"]

        return pipeline_configs, fed_optimizer_configs, auxiliary_config, automation_config, privacy_config

    @staticmethod
    def send_client_command(node, cmd, with_text_output=False, combine_stdout_stderr=True):
        """
        Opens an SSH channel to client node, executes a command, waits for completion, and returns the exit status.
        Status:     (0) success,    (1) error

        :param node: str, the node's IP address
        :param cmd: str, the command you want to send via SSH
        :param with_text_output: bool, Return received message (return format: msg (str), status (int))
        :param combine_stdout_stderr: bool, Merge stdout and stderr (default: True)
        running (optional)
        """

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            # This validates we are using a proper IP address.
            # Sometimes it happens we pass a full ssh instruction to this functions. then we need to clean the "node".
            socket.inet_aton(node)
        except OSError:
            # We need a clean hostname, i.e. IP address.
            if "@" in node:
                node = node.split("@")[1]
            else:
                pass

        client.connect(hostname=node, port=22, username="ubuntu")

        stdin, stdout, stderr = client.exec_command(command=f"{cmd}\n")
        out = stdout.readlines()

        if len(out) > 0:
            print(f"Node {node}: {out}")

        status = stdout.channel.recv_exit_status()
        client.close()

        if with_text_output:
            return out, status
        else:
            return status

    @staticmethod
    def generate_experiment_name(pipeline, ml_model, data_dist, num_clients, optimizer, inventory_name=None,
                                 dropout_rate=0, use_dp="userlevel", training_rounds=10, noise_multiplier=0,
                                 precision=16):
        """
        Generates an experiment name for an object involved in an experiment (server or client).
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")

        if inventory_name is not None:
            inventory_name = inventory_name.split("/")[-1]
            inventory_name = inventory_name.split(".")[0]
            inventory_name = inventory_name.replace("_", ".")
            inventory_name = f"_{inventory_name}"
        else:
            inventory_name = "_sim"

        if "." in str(dropout_rate):
            dropout_rate = str(dropout_rate)
            dropout_rate = dropout_rate.replace(".", "")

        # The missing underscore is intentional. See inventory_name preprocessing above.
        return f"{timestamp}_flbench{inventory_name}_{pipeline}_{ml_model}_{optimizer}_{data_dist}_{training_rounds}_" \
               f"rounds_{num_clients}_clients_{dropout_rate}_dropout_{use_dp}_{noise_multiplier}"

    def check_if_tmux_session_is_alive(self, node, target_session="flbench"):
        """
        This method checks if the session for running an experiment is still active on a remote client. It returns once
        it receives a "no server running on..." error message (str) incl. error status 1. The request is timed every 30
        seconds.
        :param node: str, IP address of a remote ssh-enabled client
        :param target_session: str, tmux session name which to check for.
        """
        cmd = f"tmux has-session -t {target_session}"
        wait_time_minutes = self.automation_config["tmux_check_alive_experiment_automation_minutes"]

        try:
            out, status = self.send_client_command(node, cmd, with_text_output=True, combine_stdout_stderr=True)
            if status == 0 and len(out) == 0:
                time.sleep(60 * wait_time_minutes)
                self.check_if_tmux_session_is_alive(node=node, target_session=target_session)
            else:
                return
        except paramiko.SSHException():
            # Required to run with very slow clients such as Raspberry Pi. Prerequisite: SSH debugging needed to happen
            # beforehand.
            msg = f"Resource {node} busy. SSH connection unavailable. Next try in {wait_time_minutes} minute(s)."
            logging.info(msg)
            print(msg)
            pass

    def update_git_repo_multithread(self, ips: list, server: str = None):
        all_nodes = []
        if server is not None:
            server = server.split(":")[0]
            all_nodes.append(server)

        # all node list
        all_nodes += ips

        self.threads_dict = {}
        # Updating the GIT repository on all clients and the server
        for node in all_nodes:
            name = f"node {node}"
            target_fn = self.update_git_on_client
            kwargs = {"node": node}
            self.create_thread(name=name, target=target_fn, kwargs=kwargs)

        # Wait for all threads to complete
        self.join_threads()

        print("> Done updating Git repositories on remote nodes.")

    def update_git_on_client(self, node):
        print(f"Starting git update on {node}")
        command = f"cd {self.remote_flbench_directory}/ && /usr/bin/git pull --recurse-submodules\n"
        status = self.send_client_command(node=node, cmd=command)

        print(f"Updated Git repository on node {node} with exit status {status}.")
        return

    def create_thread(self, name, target, kwargs):
        thread = Thread(target=target, kwargs=kwargs)
        thread.start()
        self.threads_dict[name] = thread
        return thread

    def join_threads(self):
        """
        Blocks the program to wait for all threads to complete their task, i.e. the remote clients finish their work.
        :return: None
        """
        for name, thread in self.threads_dict.items():
            thread.join()

    def set_fl_server_port(self):

        while len(self.available_ports) == 0:
            # Wait for a port to become available. This blocks the process until it can proceed.
            time.sleep(5)

        return self.available_ports.pop()