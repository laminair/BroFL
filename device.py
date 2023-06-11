import argparse
import multiprocessing
import os
import threading

import pytorch_lightning as pl
import json
import torch
import datetime
import logging

import wandb
import flwr as fl

from pathlib import Path
from utils.init_device import return_class_from_pythonic_string as load_class
from utils.logger import init_logger, get_ip_address
from modules.monitoring.monitor import HWMonitor

from collections import OrderedDict
from typing import Dict, Any

# We seed the entire setup and set the pl.Trainer to `deterministic`.
pl.seed_everything(seed=43)
logging.getLogger("lightning").setLevel(logging.ERROR)

# We use this for GPUs with Tensor Cores
torch.set_float32_matmul_precision("high")


class FLBenchDevice(fl.client.NumPyClient):

    def __init__(self, server_endpoint="127.0.0.1:8080", is_sim_client=False, use_dp=False, target_epsilon=10,
                 *args, **kwargs):
        self.server_endpoint = server_endpoint

        self.file_path = Path(os.path.dirname(os.path.realpath(__file__)))

        # This is the placeholder for all config items sent by the server.
        self.client_configured = False
        self.is_sim_client = is_sim_client

        # Will be initialized once the client receives a task.
        self.monitor = None
        self.logger = None
        self.trainer = None
        self.model = None
        self.dataloader = None

        # We maintain a local_config for local, non-federated training. This variable will be ignored for federated
        # training as the configuration is coming from the server. Clients are treated 100% ephemeral.
        self.local_config = {**vars(*args), **kwargs}

        self.project_dir = Path(os.path.dirname(os.path.realpath(__file__)))

    def get_parameters(self, config: Dict[str, Any]):
        """
        Fetches model parameters. See https://flower.dev/docs/apiref-flwr.html#flwr.client.NumPyClient.get_parameters
        for reference.
        :return: List of all local model parameters (list)
        """

        if self.client_configured is False:
            self.init_client(config=config)

        state_dict = self.model.state_dict()

        if "quantizer" in config.keys():
            # Hook for client-side federated optimization. We're not altering how the model works with data
            # here. We re-initialize the quantizer for every training round to prevent artifacts.
            assert "quantization_level" in config.keys()
            pythonic_path = f"modules.strategies.{config['quantizer']}"
            quantizer = load_class(pythonic_path)(config["quantization_level"])
            state_dict = quantizer.quantize(state_dict)

        return [val.cpu().numpy() for _, val in state_dict.items()]

    def set_parameters(self, params):
        """
        Sets the initial model parameters before starting a training round.
        See: https://flower.dev/docs/apiref-flwr.html
        :param params: model parameters
        :return: Model parameter dictionary (OrderedDict)
        """
        params_dict = zip(self.model.state_dict().keys(), params)
        new_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(new_state_dict, strict=False)

    def get_properties(self, config):
        result = {}
        for k, t in config.items():
            if k in self.__dict__:
                result[k] = self.__dict__[k]
        return result

    def fit(self, parameters, config):
        """
        Federated training process incl. extended logging.
        :param parameters: Initial model parameters (dict)
        :param config: Training configuration
        :return: model parameters (list), length_of_dataset (int), additional information (dict)
        """

        # Configures and initializes the client if not done before.
        config = self.init_client(config=config)

        # Create meta data dict (to send to server)
        meta_data_dict = {}

        self.set_parameters(parameters)

        # If we're running FedProx, we need to save the initial parameters to calculate the proximal term.
        if config["fl_optimizer"] == "fedprox":
            # We need to send the initial parameters to the ML model and set the proximal µ.
            self.model.set_mu(mu=config["proximal_mu"], init_parameters=parameters)

        # We log whether the client is set to fail for our dropout analysis
        try:
            wandb.log({"system/intended_failure": config["should_dropout"]})
        except wandb.errors.Error:
            pass

        # At this point, we are ready to start training.
        if config["should_dropout"] > 0 or config["should_dropout"] is True:
            # We return nothing as the client failed... This terminates the process immediately.
            # We need the trainloader init for the len_trainset variable to be loaded.
            self.dataloader.train_dataloader()
            return

        start_time = datetime.datetime.now()

        # Here is where the actual training is happening.
        initial_params = self.store_init_params(config=config)
        updated_params = self.start_ml_training(config=config)
        meta_data_dict = self.apply_userlevel_dp(config=config, initial_params=initial_params, updated_params=updated_params,
                                                 meta_data_dict=meta_data_dict)

        end_time = datetime.datetime.now()
        duration = end_time - start_time

        if config["logger"]["logger_type"] == "wandb" and not self.is_sim_client:
            wandb.log({"train/num_samples": len(self.dataloader.train_dataloader())})
        else:
            logging.info(f"train_num_samples: {duration}.")

        self.log_time(logger=self.logger, start_time=start_time, end_time=end_time, duration=duration, config=config,
                      is_sim_client=self.is_sim_client)

        self.dataloader.train_dataloader()
        return updated_params, self.dataloader.len_trainset, meta_data_dict

    def evaluate(self, parameters, config):
        """
        Model evaluation (test) after the model merging process. Incl. extended logging.
        :param parameters: Model parameters (list)
        :param config: Test configuration
        :return: model loss (float), testset length (int), additional information (dict)
        """
        # Configures and initializes the client if not done before. We need this here as evaluation could happen before
        # a client has ever been called for training.
        config = self.init_client(config=config)
        self.set_parameters(parameters)

        start_time = datetime.datetime.now()
        results = self.start_ml_testing()

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        try:
            loss = results[0]["test/loss"]
            accuracy = results[0]["test/accuracy"]
        except KeyError:
            loss = results[0]["test_loss"]
            accuracy = results[0]["test_accuracy"]

        self.log_time(logger=self.logger, start_time=start_time, end_time=end_time, duration=duration, config=config,
                      is_sim_client=self.is_sim_client)
        self.dataloader.test_dataloader()
        return loss, self.dataloader.len_testset, {"loss": loss, "accuracy": accuracy}

    def launch_federated(self):
        # Launch the client
        fl.client.start_numpy_client(server_address=self.server_endpoint, client=self)

    def launch_local(self):
        # We need to initialize the client for training here as well. As we do not have a server that communicates the
        # training directive, we must do it locally.
        config = self.init_client()

        # Train the model
        start_time = datetime.datetime.now()
        # You could return the model params here, if you want.
        params = self.start_ml_training(config=config)
        self.set_parameters(params=params)
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        self.log_time(logger=self.logger, start_time=start_time, end_time=end_time, duration=duration, config=config)

        # Evaluate the model
        self.start_ml_testing()

    def launch(self):
        """
        This function is called on all clients. The correct training (federated/local) is called based on the
        "data_dist" parameter
        :return:
        """
        if self.local_config["data_dist"] == "local":
            assert len(self.local_config.keys()) > 0, "Make sure to pass commandline args to the client for local " \
                                                      "training."
            self.launch_local()

        else:
            self.launch_federated()

        if not self.is_sim_client:
            # Kills the monitoring thread after completion. Do not delete this line.
            self.monitor.stop_monitor.set()
            wandb.finish()

    @staticmethod
    def eval_subprocess(trainer: pl.Trainer, model, datamodule, mp_dict):
        """
        We need this wrapper to read metrics from the PL testing function.
        """
        results = trainer.test(model=model, datamodule=datamodule)
        mp_dict.update({"results": results})

    @staticmethod
    def train_subprocess(trainer: pl.Trainer, model, datamodule, config, mp_dict):
        """
        We need this function to return the state_dict from the GPU before we kill the process.
        Otherwise, updates / training progress are lost.
        """
        trainer.fit(model=model, datamodule=datamodule)

        state_dict = model.state_dict()
        if "quantizer" in config.keys():
            # Hook for client-side federated optimization. We're not altering how the model works with data
            # here. We re-initialize the quantizer for every training round to prevent artifacts.
            assert "quantization_level" in config.keys()
            pythonic_path = f"modules.strategies.{config['quantizer']}"
            quantizer = load_class(pythonic_path)(config["quantization_level"])
            state_dict = quantizer.quantize(state_dict)

        params = [val.cpu().numpy() for _, val in state_dict.items()]
        mp_dict.update({"params": params})

    def configure_client(self, config: Dict[str, Any]) -> None:
        """
        We use this function to process the configuration received from the server.
        It typically contains the ML model architecture, the requested data distribution, and associated
        hyper-parameters.
        """
        if config["data_dist"] != "local":
            assert self.server_endpoint is not None, "For federated training you must provide a server endpoint."

    def init_ml_pipeline(self, config: Dict[str, Any]) -> None:
        """
        This function initializes a client based on the configuration received from the server.
        """
        if self.client_configured is True:
            return

        with open(f"{self.project_dir}/config.json", "r") as f:
            project_cfg = json.load(f)
            f.close()

        self.pipeline_config = project_cfg["pipelines"][config["pipeline"]]
        # model_kwargs = {**config["hparams"] **config["privacy"]}
        self.model = load_class(self.pipeline_config["models"][config["ml_model"]])(**config["hparams"])
        # privacy_engine=self.privacy_engine)
        self.dataloader = load_class(self.pipeline_config["dataloader"]["path"])(**config["hparams"])

    def init_logger(self, config: Dict[str, Any]) -> None:
        """
        This function initializes the logger and asynchronous hardware monitoring.
        """
        # Logging capabilities
        print("Logger active on client")
        self.logger = init_logger(logger=config["logger"]["logger_type"],
                                  experiment_name=config["experiment_name"],
                                  project_name=config["logger"]["project_name"],
                                  project_group=config["logger"]["project_group"],
                                  logger_suffix=config["logger"]["logger_suffix"],
                                  wandb_entity=config["logger"]["wandb_entity"]
                                  )

        # Hardware Monitoring v2
        if type(self.server_endpoint) is str:
            server_ip = self.server_endpoint.split(":")
            proc_ip_addrs = [server_ip]
        else:
            proc_ip_addrs = None

        if not config["is_sim_client"]:
            self.monitor = HWMonitor(monitoring_freq=config["logger"]["logging_freq"], proc_ip_addrs=proc_ip_addrs,
                                     use_scapy=config["logger"]["use_scapy_network_monitoring"], logger=self.logger,
                                     stop_event=threading.Event())

            # Starts the async hardware monitoring thread
            self.monitor.start()

    def init_lightning_trainer(self, config) -> None:

        # assert self.logger is not None

        progbar = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

        try:
            mps = torch.backends.mps.is_available()
        except:
            mps = False

        # Determine the device accelerator
        if config["logger"]["force_accelerator_type"]:
            accelerator = config["logger"]["force_accelerator_type"]
            devices = 1
        elif torch.cuda.is_available():
            accelerator = "gpu"
            devices = torch.cuda.device_count()
        elif mps:
            accelerator = "mps"
            devices = 1
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        else:
            accelerator = "cpu"
            devices = 1

        self.trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            deterministic=True,
            logger=self.logger,
            log_every_n_steps=config["logger"]["logging_freq"],
            min_epochs=config["logger"]["min_local_epochs"],
            max_epochs=config["logger"]["max_local_epochs"],
            fast_dev_run=config["logger"]["fast_dev_run"],
            callbacks=[progbar],  # , dp_callback
            precision=config["trainer"]["precision"]
        )

    def init_client(self, config=None) -> dict:
        """
        The client needs to be configured when called for the first time. Configuration happens depending on the server.
        :param config: Configuration parameters coming from the server. (dict)
        """

        config = self.build_config(server_config=config, cmdline_config=self.local_config)
        self.configure_client(config=config)
        # self.configure_privacy_accountant(config)
        self.init_ml_pipeline(config=config)
        if not self.is_sim_client:
            self.init_logger(config=config)

        self.client_configured = True

        # We reinitialize the trainer for every FL round we train. Otherwise, the client would just send its parameters
        # from the last training round.
        self.init_lightning_trainer(config=config)

        return config

    def build_config(self, server_config: dict = None, cmdline_config: dict = None) -> dict:
        """
        Here we build the client config.
        The config received from the server is treated superior to the local config.
        When running clients in an FL system, clients are treated ephemerally, i.e. all configuration is sent by the
        server.
        When running local baseline experiments, you must configure the client yourself as there is no server, which may
        send a configuration.
        """
        with open(f"{self.project_dir}/config.json", "r") as f:
            project_cfg = json.load(f)
            f.close()

        logger_config = project_cfg["auxiliary"]
        # privacy_config = project_cfg["privacy_techniques"]["differential_privacy"]["parameter_configurations"]

        # Merge local config with server-side config
        # This is especially relevant for DP workloads
        config = {**cmdline_config}
        config.update(**config["kwargs"])
        del config["kwargs"]
        del config["args"]
        del config["self"]

        if server_config is not None:
            config.update(**server_config)

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

        # We need to structure hyperparameters
        config["hparams"] = {
            "client_id": config["client_id"],
            "data_dist": config["data_dist"]
        }

        for param, val in config.items():
            if "hparam_" in param:
                name = param.replace("hparam_", "")
                config["hparams"][name] = val

        try:
            if self.local_config["data_dist"] == "local":
                config["hparams"].update({
                    **project_cfg["pipelines"][config["pipeline"]]["hparams"][config["ml_model"]]
                })
        except KeyError:
            pass

        config["fl_optimizer"] = config["fl_optimizer"] if "fl_optimizer" in config.keys() else None

        # Privacy configuration
        config["privacy"] = {"use_dp": config["use_dp"]}
        if config["privacy"]["use_dp"]:
            # The following options will only take effect in training (`fit()`).
            # For testing / evaluation tasks we will not need any DP procedure.
            config["privacy"].update({
                    "dpfedavg_clip_norm": config["dpfedavg_clip_norm"] if "dpfedavg_clip_norm" in config.keys() else 0.0,
                    "dpfedavg_adaptive_clip_enabled": config["dpfedavg_adaptive_clip_enabled"] if "dpfedavg_adaptive_clip_enabled" in config.keys() else False,
                    "dpfedavg_noise_stddev": config["dpfedavg_noise_stddev"] if "dpfedavg_noise_stddev" in config.keys() else 0.0,
                    "dp_noise_multiplier": config["dp_noise_multiplier"] if "dp_noise_multiplier" in config.keys() else 0.0
            })

            for key in config["privacy"].keys():
                try:
                    del config[key]
                except KeyError:
                    pass

        # Logger configuration
        config["logger"] = {
            "logger_type": logger_config["logger"],
            "wandb_entity": logger_config["wandb_entity"],
            "project_name": logger_config["project_name"],
            "project_group": "_".join(config['experiment_name'].split("_")[3:]),
            "logging_freq": logger_config["logging_freq"],
            "logger_suffix": logger_config["logger_suffix"],
            "min_local_epochs": logger_config["min_local_epochs"],
            "max_local_epochs": logger_config["max_local_epochs"],
            "seed": logger_config["seed"],
            "use_scapy_network_monitoring": logger_config["use_scapy_network_monitoring"],
            "fast_dev_run": logger_config["fast_dev_run"],
            "force_accelerator_type": logger_config["force_accelerator_type"],
        }

        for key in config["logger"].keys():
            try:
                del config[key]
            except KeyError:
                pass

        config["experiment_name"] = f"{config['experiment_name']}_{get_ip_address()}_client_{config['client_id'] + 1}"

        # with open(f"{self.file_path}/conf_device_{config['hparams']['client_id']}.txt", "w") as f:
        #     print(config, file=f)
        #     f.close()

        return config

    def start_ml_training(self, config):
        mgr = multiprocessing.Manager()
        mp_dict = mgr.dict()
        proc = multiprocessing.Process(
            target=self.train_subprocess,
            kwargs={
                "trainer": self.trainer,
                "model": self.model,
                "datamodule": self.dataloader,
                "config": config,
                "mp_dict": mp_dict
            })
        proc.start()
        proc.join()
        try:
            proc.close()
        except ValueError as e:
            print(f"Could not close Fit Process. {e}")

        params = mp_dict["params"]
        del (mgr, mp_dict, proc)
        return params

    def start_ml_testing(self):
        mgr = multiprocessing.Manager()
        mp_dict = mgr.dict()
        proc = multiprocessing.Process(
            target=self.eval_subprocess,
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
            print(f"Could not close Test Process. {e}")
        results = mp_dict
        results = results["results"]
        del (mgr, mp_dict, proc)
        return results

    def store_init_params(self, config):
        # Init params will only be stored when running DP workloads.
        if config["privacy"]["use_dp"]:
            return {key: value.clone().detach() for key, value in self.model.state_dict().items()}
        else:
            return None

    def apply_userlevel_dp(self, config, initial_params, updated_params, meta_data_dict):
        if config["privacy"]["use_dp"]:
            self.set_parameters(updated_params)
            updated_params = self.model.state_dict()
            # Here we calculate the L2 weight update norm and return an iterator bit if
            diff = 0
            for key in initial_params.keys():
                diff += torch.sum(torch.abs(initial_params[key] - updated_params[key]))

            update_norm = diff.item()
            meta_data_dict.update({
                "dpfedavg_norm_bit": True if update_norm <= config["privacy"]["dpfedavg_clip_norm"] else False
            })

            return meta_data_dict
        else:
            return meta_data_dict

    @staticmethod
    def log_time(logger, start_time, end_time, duration, config, is_sim_client=False):
        if config["logger"]["logger_type"] == "wandb" and not is_sim_client:
            # Log the actual training time
            logger.log_metrics({"timing/training_time": duration.total_seconds()})  # Experiment duration in seconds
            logger.log_metrics({"timing/training_start_time": start_time.timestamp()})
            logger.log_metrics({"timing/training_end_time": end_time.timestamp()})
        else:
            logging.info(f"training_time: {duration}.")
            logging.info(f"training_start_time: {start_time.timestamp()}.")
            logging.info(f"training_end_time: {end_time.timestamp()}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLBench client device configuration.")
    parser.add_argument("--server-endpoint", type=str, required=False)

    # All following commandline arguments are only required for local training. They will be ignored in federated
    # settings as the server sends a config.
    parser.add_argument("--client-id", type=int, required=False)
    parser.add_argument("--pipeline", type=str, required=False)
    parser.add_argument("--ml-model", type=str, required=False)
    parser.add_argument("--data-dist", type=str, required=False)
    parser.add_argument("--use-dp", type=bool, required=False, default=False)
    parser.add_argument("--noise-multiplier", type=int, required=False, default=10)
    parser.add_argument("--experiment-name", type=str, required=False, default="test")

    args = parser.parse_args()

    # See: https://github.com/pytorch/pytorch/issues/3492
    multiprocessing.set_start_method("spawn", force=True)
    device = FLBenchDevice(**vars(args))

    device.launch()
