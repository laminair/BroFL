import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.common.dp import add_gaussian_noise
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy
from opacus.accountants.rdp import RDPAccountant, privacy_analysis
from opacus.accountants.analysis.rdp import compute_rdp

import copy


class DPFixedClipping(Strategy):
    """
    Wrapper for configuring a Strategy for DP with Fixed Clipping.
    DP-FedAvg [McMahan et al., 2018] strategy.
    Paper: https://arxiv.org/pdf/1710.06963.pdf
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(self, strategy: Strategy, available_clients: int, num_sampled_clients: int, clip_norm: float,
                 noise_multiplier: float = 1, server_side_noising: bool = True, delta: float = 0.01, logger=None,
                 dataset_name=None, *args, **kwargs) -> None:
        super().__init__()
        self.strategy = strategy
        # Doing fixed-size subsampling as in https://arxiv.org/abs/1905.03871.
        self.num_sampled_clients = num_sampled_clients

        if clip_norm <= 0:
            raise Exception("The clipping threshold should be a positive value.")
        self.clip_norm = clip_norm

        if noise_multiplier < 0:
            raise Exception("The noise multiplier should be a non-negative value.")
        self.noise_multiplier = noise_multiplier

        self.server_side_noising = server_side_noising
        self.available_clients = available_clients
        self.privacy_accountant = RDPAccountant()
        self.logger = logger
        self.dataset_name = dataset_name

        # We set delta depending on the number of samples in the dataset.
        if self.dataset_name == "blond":
            self.delta = 1 / 10_542
        elif self.dataset_name == "mnist":
            self.delta = 1 / 628_110
        elif self.dataset_name == "shakespeare":
            self.delta = 1 / 3_380_891
        else:
            self.delta = delta

    def __repr__(self) -> str:
        rep = "Strategy with DP with Fixed Clipping enabled."
        return rep

    def _calc_client_noise_stddev(self) -> float:
        return float(self.noise_multiplier * self.clip_norm / (self.num_sampled_clients ** 0.5))

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        additional_config = {"dpfedavg_clip_norm": self.clip_norm}
        if not self.server_side_noising:
            additional_config[
                "dpfedavg_noise_stddev"
            ] = self._calc_client_noise_stddev()

        client_list = self.strategy.configure_fit(
            server_round, parameters, client_manager
        )

        for _, fit_ins in client_list:
            fit_ins.config.update(additional_config)

        return client_list

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            return None, {}

        # For the eps calculation, we need the number of samples returned by a client
        eps, sample_rate = self.calculate_epsilon(results=results, server_round=server_round, delta=self.delta,
                                                  noise_multiplier=self.noise_multiplier)

        # Forcing unweighted aggregation, as in https://arxiv.org/abs/1905.03871.
        for _, fit_res in results:
            fit_res.num_examples = 1
            fit_res.parameters = ndarrays_to_parameters(
                add_gaussian_noise(
                    parameters_to_ndarrays(fit_res.parameters),
                    self._calc_client_noise_stddev(),
                )
            )

        if self.logger is not None:
            self.logger.log_metrics({
                "dp/epsilon": eps,
                "dp/delta": self.delta,
                "dp/sample_rate": sample_rate,
                "dp/clip_norm": self.clip_norm,
                "dp/noise_multiplier": self.noise_multiplier
            })

        return self.strategy.aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.strategy.evaluate(server_round, parameters)

    @staticmethod
    def calculate_epsilon(results, server_round, delta, noise_multiplier):
        """
        We use PyTorch opacus RDP privacy accounting to observe our epsilon.
        Delta is an external control parameter and set in line with
        """
        # We run an accountant step with the calculated noise_multiplier and the fixed sampling rate.
        # With a step, we generate the RDP alphas that we later use to calculate the privacy budget (eps)

        processed_samples = 0
        for _, fit_res in results:
            processed_samples += fit_res.num_examples

        sample_rate = 1 / processed_samples
        orders = [i for i in range(2, 33)]
        rdp = compute_rdp(q=sample_rate, noise_multiplier=noise_multiplier, steps=(server_round + 1), orders=orders)
        eps, best_alpha = privacy_analysis.get_privacy_spent(orders=orders, rdp=rdp, delta=delta)
        return eps, sample_rate


class DPAdaptiveClippingWrapper(DPFixedClipping):
    """
    Wrapper for configuring a Strategy for DP with Adaptive Clipping.
    DP-FedAvg [Andrew et al., 2019] with adaptive clipping.
    Paper: https://arxiv.org/pdf/1905.03871.pdf
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(self, strategy: Strategy, available_clients: int, num_sampled_clients: int, clip_norm_lr: float = 0.2,
                 noise_multiplier: float = 1, server_side_noising: bool = True, init_clip_norm: float = 0.1,
                 clip_norm_target_quantile: float = 0.5, clip_count_stddev: Optional[float] = None,
                 delta: float = 0.0001, logger=None, dataset_name=None) -> None:
        super().__init__(
            strategy=strategy,
            num_sampled_clients=num_sampled_clients,
            clip_norm=init_clip_norm,
            noise_multiplier=noise_multiplier,
            server_side_noising=server_side_noising,
            available_clients=available_clients,
            delta=delta,
            logger=logger,
            dataset_name=dataset_name
        )
        self.clip_norm_lr = clip_norm_lr
        self.clip_norm_target_quantile = clip_norm_target_quantile
        self.clip_count_stddev = clip_count_stddev
        if self.clip_count_stddev is None:
            self.clip_count_stddev = 0
            if noise_multiplier > 0:
                # We have a small system. Therefore, we need to set the stddev denominator lower. Default: 20
                # See McMahan et al. on how to set the denominator. In accordance with Andrew et al. we aim for a
                # denominator of ~5 (=> Andrew et al. m = 100 clients per training round with σ(b) = m / 20)
                denominator = 2 if self.num_sampled_clients < 20 else 20
                self.clip_count_stddev = self.num_sampled_clients / denominator

        if noise_multiplier:
            self.noise_multiplier = (self.noise_multiplier ** (-2) + (2 * self.clip_count_stddev) ** (-2)) ** (-0.5)

    def __repr__(self) -> str:
        rep = "Strategy with DP with Adaptive Clipping enabled."
        return rep

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        additional_config = {"dpfedavg_adaptive_clip_enabled": True, "dpfedavg_clip_norm": self.clip_norm}

        client_list = super().configure_fit(
            server_round, parameters, client_manager
        )

        for _, fit_ins in client_list:
            fit_ins.config.update(additional_config)

        return client_list

    def _update_clip_norm(self, results: List[Tuple[ClientProxy, FitRes]]) -> None:
        # Calculating number of clients which set the norm indicator bit
        norm_bit_set_count = 0
        for client_proxy, fit_res in results:
            if "dpfedavg_norm_bit" not in fit_res.metrics:
                raise Exception(
                    f"Indicator bit not returned by client with id {client_proxy.cid}."
                )
            if fit_res.metrics["dpfedavg_norm_bit"]:
                norm_bit_set_count += 1
        # Noising the count
        noised_norm_bit_set_count = float(np.random.normal(norm_bit_set_count, self.clip_count_stddev))

        noised_norm_bit_set_fraction = noised_norm_bit_set_count / len(results)
        # Geometric update
        self.clip_norm *= math.exp(-self.clip_norm_lr * (noised_norm_bit_set_fraction - self.clip_norm_target_quantile))
        if self.logger is not None:
            self.logger.log_metrics({
                "dp/clip_norm": self.clip_norm,
            })

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            return None, {}
        new_global_model = super().aggregate_fit(server_round, results, failures)
        self._update_clip_norm(results)

        if self.logger is not None:
            self.logger.log_metrics({
                "dp/adaptive_clip_norm": self.clip_norm,
            })

        return new_global_model
