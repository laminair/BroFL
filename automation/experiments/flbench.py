"""
This script automates all FLBench experiments and ensures reproducibility.
"""
import argparse
import datetime
import json
import time
from pathlib import Path
from utils import ExperimentAutomatorBase


class ExperimentAutomator(ExperimentAutomatorBase):

    def __init__(self, inventory_path, server_endpoint="127.0.0.1:8080", experiment_plan_path=None, dryrun=False):
        super(ExperimentAutomator, self).__init__(inventory_path, server_endpoint, experiment_plan_path, dryrun)

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

                if experiment["devices"] > 1 and experiment["data_dist"] != "local":
                    self.configure_and_launch_server(
                        pipeline=experiment["pipeline"],
                        data_dist=experiment["data_dist"],
                        num_devices=experiment["devices"],
                        optimizer=experiment["optimizer"],
                        ml_model=experiment["model"],
                        experiment_name=experiment_name,
                        use_dp=experiment["use_dp"],
                        noise_multiplier=experiment["noise_multiplier"],
                        training_rounds=experiment["training_rounds"],
                        client_dropout_rate=experiment["dropout_rate"]
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
                    time.sleep(0.3)

                time.sleep(30)  # Wait 30sec until the next experiment
                self.join_threads()
                experiment_counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLBench experiment automation configuration.")
    parser.add_argument("--inventory-path",
                        type=str,
                        required=False,
                        default="/opt/flbench/automation/ansible/inventory_static.cfg")
    parser.add_argument("--server-endpoint",
                        type=str,
                        required=False,
                        default="127.0.0.1:8080",
                        help="Specify the ip and port, e.g. 127.0.0.1:8080")
    parser.add_argument("--dryrun",
                        type=bool,
                        required=False,
                        default=False,
                        help="Use this to generate the experiment plan only.")
    parser.add_argument("--experiment-plan-path",
                        type=str,
                        required=False,
                        help="Use this to generate the experiment plan only.")

    args = parser.parse_args()

    automator = ExperimentAutomator(inventory_path=args.inventory_path, server_endpoint=args.server_endpoint,
                                    dryrun=args.dryrun)
    automator.launch()
