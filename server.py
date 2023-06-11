import datetime
import threading

import os
import flwr as fl
import pytorch_lightning as pl
import torch

import json
import argparse
import random
from collections import OrderedDict
import logging
import multiprocessing

from pathlib import Path

from flwr.common import ndarrays_to_parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from typing import Dict

from utils.logger import init_logger, get_ip_address
from utils.init_device import return_class_from_pythonic_string as load_class
from modules.monitoring.monitor import HWMonitor
from modules.client_managers import MemoryClientManager
from modules.strategies.dp_wrapper import DPAdaptiveClippingWrapper
from opacus import PrivacyEngine


# We seed the entire setup and set the pl.Trainer to `deterministic`.
pl.seed_everything(seed=43)
logging.getLogger("lightning").setLevel(logging.ERROR)

# We use this for GPUs with Tensor Cores
torch.set_float32_matmul_precision("high")


class FLBenchServer(object):
    """
    This class implements a Federated Learning server for deployment in distributed systems and for simulating
    large-scale FL systems on a single machine. The server is self-contained, i.e., all you need in flower and pytorch
    on your machine.
    """

    def __init__(self, *args, **kwargs):
        """
        All string parameters are directly linked to the config.json file, i.e., all information is laoded from there.
        :param pipeline: The dataset you want to train on (str)
        :param ml_model: The ML model architecture you want to use (str)
        :param server_endpoint: The server's IP address & port number (str).
        :param n_clients: The number of clients the server should expect to find in the system (int)
        :param fl_strategy: The FL optimizer to use for a workload
        :param experiment_name: An optional text to identify experiments more easily (str)
        :param client_dropout_rate: The probability of how may clients should drop out in a single FL round (float; 0.0 - 1.0)
        :param training_rounds: The number of FL training rounds (int)
        :param is_simulation: Flag for simulating an entire FL system on a local machine (bool)
        :param use_dp: Flag for using differential privacy in experiments (bool)
        :param dp_noise_multiplier: The control variable for noise in differentially private workloads (float)

        Please note that for the epsilon to take effect you may need to tune delta (the data leakage probability).
        Per default delta is set to 0.01. We use epsilon to control the information flow.
        In a production system you rather want to have a tight delta < 0.001, depending on the amount of data you have.
        """

        # Here we build our config
        self.cmdline_config = vars(*args)
        self.cmdline_config.update({**kwargs})

        self.file_path = Path(os.path.dirname(os.path.realpath(__file__)))

        # We configure the Flower Server and Device Monitoring.
        config = self.build_config()
        # self.configure_privacy_accountant(config)
        self.init_logger(config=config)
        config = self.init_server(config=config)
        self.init_pt_lightning_trainer(config=config)
        self.start_time = datetime.datetime.now()
        try:
            self.is_simulation = config["is_simulation"]
        except KeyError:
            self.is_simulation = False

    def launch(self, *args, **kwargs):
        # Start server for n rounds with a timeout of 7200 seconds (= 2hrs).
        self.start_time = datetime.datetime.now()
        server = self.cmdline_config["server_endpoint"]
        fl.server.start_server(server_address=f"{server.split(':')[0]}:{server.split(':')[1]}",
                               config=self.server_config,
                               strategy=self.strategy,
                               client_manager=MemoryClientManager()
                               )
        end_time = datetime.datetime.now()
        duration = end_time - self.start_time

        # Kills the monitoring thread
        self.monitor.stop_monitor.set()
        self.logger.log_metrics({"timing/total_time": duration.total_seconds()})

    def server_side_evaluation_fn(self, model: pl.LightningModule):
        """
        This function evaluates the global model against the "local" test dataset
        """

        def evaluate(server_round: int, params, config):
            # This ultimately is the "set_parameters()" functions on client-side.
            params_dict = zip(self.model.state_dict().keys(), params)
            new_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(new_state_dict, strict=False)

            # Time to first server-side test
            ramp_up_time = datetime.datetime.now() - self.start_time

            # if self.is_simulation:
            self.client_instructions.update({"is_sim_client": True})
            mgr = multiprocessing.Manager()
            mp_dict = mgr.dict()
            proc = multiprocessing.Process(
                target=self.eval_simulation,
                kwargs={
                    "trainer": self.trainer,
                    "model": self.model,
                    "datamodule": self.dataloader,
                    "mp_dict": mp_dict
                })
            proc.start()
            proc.join()
            try:
                proc.close()
            except ValueError as e:
                print(f"Could not close Evaluation Process. {e}")
            results = mp_dict
            results = results["results"]
            del (mgr, mp_dict, proc)

            # Timing logger
            end_time = datetime.datetime.now()
            duration = end_time - self.start_time
            self.logger.log_metrics({
                "global_performance/loss": results[0]['test/loss'],
                "global_performance/accuracy": results[0]['test/accuracy'],
                "global_performance/f1_score": results[0]['test/f1_score'],
                "timing/time_to_first_test_sample": ramp_up_time.total_seconds(),
                "timing/evaluation_time": duration.total_seconds()
            })

            return results[0]["test/loss"], {"accuracy": results[0]["test/accuracy"],
                                             "f1_score": results[0]["test/f1_score"]}

        return evaluate

    def on_fit_config_fn(self, client: ClientProxy, client_manager: MemoryClientManager,
                         server_round: int) -> Dict[str, Scalar]:

        client_id = client_manager.client_memory[client.cid]['client_id']
        if self.client_instructions["use_dp"]:
            # We cannot run DP workloads with client dropouts as this would undermine the mathematical privacy guarantee
            is_dropout = 0
        else:
            try:
                is_dropout = self.dropout_list[server_round][client_id]
            except IndexError:
                is_dropout = 0

        # We can only use serializable dict items as we will send the client config via gRPC
        # The configuration is prepared in build_config().
        client_config = {
            "client_id": client_id,
            "should_dropout": is_dropout,
            **self.client_instructions
        }

        # with open(f"{self.file_path}/conf_server_{client_id}.txt", "w") as f:
        #     print(client_config, file=f)
        #     f.close()

        return client_config

    def init_server(self, config):
        model_config = config["model"]
        privacy_config = config["privacy"]

        # ML Model & Dataloader init
        self.model = load_class(model_config["model_path"])(**model_config["hparams"], **privacy_config)
        self.dataloader = load_class(model_config["dataloader_path"])(**model_config["dataloader_hparams"])

        # We can only set the server-side evaluation function once we have initialized our model.
        # Initialize the FL strategy
        strategy_config = config["server"]["hparams"].copy()
        strategy_config.update({
            f"{self.eval_func_name}": self.server_side_evaluation_fn(model=self.model),
            "initial_parameters": ndarrays_to_parameters(
                [val.cpu().numpy() for _, val in self.model.state_dict().copy().items()]
            )
        })

        self.strategy = load_class(config["server"]["strategy"])(**strategy_config)
        # Conditional on config["use_dp"]
        self.make_strategy_dp(config=config)

        # Server init config with a 1hr round timeout.
        self.server_config = fl.server.ServerConfig(num_rounds=config["training_rounds"], round_timeout=60 * 60 * 1)

        # Load the dropout list
        if not config["use_dp"]:
            self.dropout_list = self.configure_client_dropouts(config)

        # We need to return the config as we had to add the server-side evaluation function in this step.
        return config

    def build_config(self):

        project_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        with open(f"{project_dir}/config.json", "r") as f:
            project_cfg = json.load(f)
            f.close()

        config = self.cmdline_config.copy()
        config.update(**config["kwargs"])
        del config["kwargs"]
        del config["args"]
        del config["self"]

        # Configure version compatibility between Flower 0.18 and 1.0+
        if fl.__version__ < "1.0.0":
            self.eval_func_name = "eval_fn"
            min_eval_clients_name = "min_eval_clients"
            frac_eval_name = "fraction_eval"
            on_eval_conf_name = "on_evaluate_config_fn"
        else:
            self.eval_func_name = "evaluate_fn"
            min_eval_clients_name = "min_evaluate_clients"
            frac_eval_name = "fraction_evaluate"
            on_eval_conf_name = "on_evaluate_config_fn"

        # Add flower configuration
        strategy = config["fl_strategy"]

        config.update({
            "server": {
                "strategy": project_cfg["federated_optimizer"][strategy]["strategy"],
                "hparams": {
                    "min_available_clients": config["n_clients"],
                    "fraction_fit": project_cfg["federated_optimizer"][strategy]["params"]["fraction_fit"],
                    f"{frac_eval_name}": project_cfg["federated_optimizer"][strategy]["params"]["fraction_evaluate"],
                    # Here we would have defined the server-side evaluation function but we need to init the model first
                    "on_fit_config_fn": self.on_fit_config_fn,
                    f"{on_eval_conf_name}": self.on_fit_config_fn,   # That's intended that we use the same function.
                    # "fit_metrics_aggregation_fn": self.fit_metrics_aggregation_fn,
                    # "evaluate_metrics_aggregation_fn": self.evaluate_metrics_aggregation_fn,
                    "initial_parameters": None,

                }
            }
        })

        if strategy == "qfedavg":
            if config["pipeline"] == "shakespeare":
                q = 0.001
            else:
                q = 1

            config["server"]["hparams"].update({"q_param": q})

        if strategy == "fedprox":
            if config["pipeline"] == "blond":
                prox_mu = 1
            elif config["pipeline"] == "shakespeare":
                prox_mu = 0.001
            else:
                prox_mu = project_cfg["federated_optimizer"][strategy]["params"]["proximal_mu"]

            config["server"]["hparams"].update({"proximal_mu": prox_mu})

        # Add ML model configuration
        config.update({
            "model": {
                "model_path": project_cfg["pipelines"][config["pipeline"]]["models"][config["ml_model"]],
                "hparams": project_cfg["pipelines"][config["pipeline"]]["hparams"][config["ml_model"]],
                "dataloader_path": project_cfg["pipelines"][config["pipeline"]]["dataloader"]["path"],
                # We don't distinguish between model and dataloader hyperparameters.
                "dataloader_hparams": project_cfg["pipelines"][config["pipeline"]]["hparams"][config["ml_model"]]
            }
        })

        # Add lightning trainer configuration
        config.update({
            "trainer": {
                "min_local_epochs": project_cfg["auxiliary"]["min_local_epochs"],
                "max_local_epochs": project_cfg["auxiliary"]["max_local_epochs"],
                "fast_dev_run": project_cfg["auxiliary"]["fast_dev_run"],
                "force_accelerator_type": project_cfg["auxiliary"]["force_accelerator_type"],
                "precision": project_cfg["auxiliary"]["training_float_precision"]
            }
        })

        # Logger configuration
        config.update({
            "logger": {
                "logger_type": project_cfg["auxiliary"]["logger"],
                "wandb_project_name": project_cfg["auxiliary"]["project_name"],
                "wandb_entity": project_cfg["auxiliary"]["wandb_entity"],
                "project_group": "_".join(config['experiment_name'].split("_")[3:]),
                "experiment_name": config["experiment_name"],
                "logger_suffix": project_cfg["auxiliary"]["logger_suffix"],
                "logging_freq": project_cfg["auxiliary"]["logging_freq"],
                "use_scapy_network_monitoring": project_cfg["auxiliary"]["use_scapy_network_monitoring"]
            }
        })

        self.logger_type = config["logger"]["logger_type"]

        # Finally, we need to define what configuration we want to send to the clients.
        # In addition, we need to tag all hyperparameters.
        self.client_instructions = {
            "pipeline": config["pipeline"],
            "fl_optimizer": config["server"]["strategy"],
            "ml_model": config["ml_model"],
            "data_dist": config["data_dist"],
            "use_dp": config["use_dp"],
            "dp_noise_multiplier": config["dp_noise_multiplier"],
            "num_epochs": project_cfg["auxiliary"]["max_local_epochs"],  # Needed for DP.
            "experiment_name": config["experiment_name"],
        }

        # We need to reconfigure the differential privacy parameters.
        # Important note: If you want to use multiple deltas or alike you must change the logic here.
        config.update({
            "privacy": {
                "use_dp": config["use_dp"],
                "num_epochs": project_cfg["auxiliary"]["max_local_epochs"],
                "dp_noise_multiplier": config["dp_noise_multiplier"]
            }
        })

        for param, value in project_cfg["privacy_techniques"]["differential_privacy"]["user_level"].items():
            if type(value) == list:
                config["privacy"].update({param: value[0]})
            else:
                config["privacy"].update({param: value})

        # Here we add the model & dataloader hyperparameters. Please note that we do only maintain a single list of
        # of hyperparameters for both. If you want separate ones
        for hparam, value in project_cfg["pipelines"][config["pipeline"]]["hparams"][config["ml_model"]].items():
            self.client_instructions.update({f"hparam_{hparam}": value})

        return config

    @staticmethod
    def load_server_params(model):
        # For FedOpt methods we need to initialize the adaptive optimizer with the initial state of our model.
        # The FedOpt strategies use zip() to match existing_weights with updates. Therefore, the model's weight shapes
        # are required as initial weights.
        initial_weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
        return ndarrays_to_parameters(initial_weights)

    def init_logger(self, config):

        logger_config = config["logger"]
        self.logger = init_logger(logger=logger_config["logger_type"],
                                  wandb_entity=logger_config["wandb_entity"],
                                  project_name=logger_config["wandb_project_name"],
                                  project_group=logger_config["project_group"],
                                  experiment_name=f"{logger_config['experiment_name']}_{get_ip_address()}_server",
                                  logger_suffix=logger_config["logger_suffix"]
                                  )

        # Hardware Monitoring v2
        self.monitor = HWMonitor(logger=self.logger, monitoring_freq=logger_config["logging_freq"], proc_ip_addrs=None,
                                 use_scapy=logger_config["use_scapy_network_monitoring"], stop_event=threading.Event())

        # Starts the async hardware monitoring thread
        self.monitor.start()

    def init_pt_lightning_trainer(self, config):

        trainer_config = config["trainer"]

        # Determine the device accelerator
        if trainer_config["force_accelerator_type"]:
            accelerator = trainer_config["force_accelerator_type"]
            devices = 1
        elif torch.cuda.is_available():
            accelerator = "gpu"
            devices = torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            devices = 1
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        else:
            accelerator = "cpu"
            devices = 1

        # Set up the PyTorch Lightning Trainer
        progbar = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)
        self.trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            deterministic=True,
            logger=self.logger,
            log_every_n_steps=config["logger"]["logging_freq"],
            callbacks=[progbar],
            precision=config["trainer"]["precision"]
        )

    @staticmethod
    def configure_client_dropouts(config) -> Dict[int, list]:
        """
        Based on the number of clients we create a randomly shuffled list of dropout indicators.
        We use this list to determine the clients that should not send an update in a given round.
        """
        dropout_dict = {}
        dropout_threshold = config["client_dropouts"]
        n_clients = config["n_clients"]

        dropout_index = [1 if idx < dropout_threshold else 0 for idx in range(n_clients)]
        for rnd in range(config["training_rounds"]):
            random.shuffle(dropout_index)
            # Flower starts FL round increments at 1, not 0.
            dropout_dict[rnd + 1] = dropout_index

        return dropout_dict

    @staticmethod
    def eval_simulation(trainer: pl.Trainer, model, datamodule, mp_dict):
        """
        We need this wrapper to read metrics from the PL testing function.
        """
        results = trainer.test(model=model, datamodule=datamodule)
        mp_dict.update({"results": results})

    def make_strategy_dp(self, config):
        if config["use_dp"]:
            server_hparams_config = config["server"]["hparams"]
            num_clients = int(server_hparams_config["min_available_clients"] * server_hparams_config["fraction_fit"])
            self.privacy_engine = PrivacyEngine()
            self.strategy = DPAdaptiveClippingWrapper(
                strategy=self.strategy,
                num_sampled_clients=num_clients,
                init_clip_norm=config["privacy"]["init_clip_norm"],
                noise_multiplier=config["privacy"]["dp_noise_multiplier"],
                server_side_noising=config["privacy"]["server_side_noising"],
                clip_norm_lr=config["privacy"]["clip_norm_lr"],
                clip_norm_target_quantile=config["privacy"]["clip_norm_target_quantile"],
                clip_count_stddev=None,
                available_clients=config["server"]["hparams"]["min_available_clients"],
                delta=config["privacy"]["delta"],
                logger=self.logger,  # We need the logger to log epsilon, noise_multiplier, and delta.
                dataset_name=config["pipeline"]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLBench server configuration.")
    parser.add_argument("--pipeline", type=str, required=True)
    parser.add_argument("--ml-model", type=str, required=True)
    parser.add_argument("--data-dist", type=str, required=True, default="local")
    parser.add_argument("--use-dp", type=bool, required=False, default=False)
    parser.add_argument("--dp-noise-multiplier", type=float, required=False, default=1.0)
    parser.add_argument("--server-endpoint", type=str, required=True)
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--training-rounds", type=int, required=True, default=10)
    parser.add_argument("--client-dropouts", type=float, required=True, default=0)
    parser.add_argument("--fl-strategy", type=str, required=False, default="fedavg")
    parser.add_argument("--experiment-name", type=str, required=False, default="test")
    parser.add_argument("--is-simulation", type=bool, required=False, default=False)

    args = parser.parse_args()

    # See: https://github.com/pytorch/pytorch/issues/3492
    multiprocessing.set_start_method("spawn", force=True)
    server = FLBenchServer(
        pipeline=args.pipeline,
        ml_model=args.ml_model,
        data_dist=args.data_dist,
        use_dp=args.use_dp,
        dp_noise_multiplier=args.dp_noise_multiplier,
        server_endpoint=args.server_endpoint,
        n_clients=args.num_clients,
        training_rounds=args.training_rounds,
        client_dropouts=args.client_dropouts,
        fl_strategy=args.fl_strategy,
        experiment_name=args.experiment_name,
        is_simulation=args.is_simulation
    )

    server.launch()
