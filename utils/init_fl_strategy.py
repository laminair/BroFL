import flwr as fl


def init_fl_strategy(strategy, client_frac, min_available_clients, client_frac_eval=None, eval_fn=None, model=None,
                     server_learning_rate=None, server_momentum=None, eta=0.1, eta_l=0.1, beta_1=None, beta_2=None,
                     tau=1e-09, *args, **kwargs):
    """
    This initiates a FL strategy.
    :param strategy: FL strategy name. Available choices: ["FedAvg", "FedMA", "FedSGD", "FedAvgM", "QFedAvg", "FaultTolerantFedAvg", "FedOpt", "FedAdam"] (str)
    :param client_frac: Relative fraction of clients to sample in a single training round. (float; 0.0 - 1.0)
    :param min_available_clients: Minimum number of clients that need to be available to start a training process. (int)
    :param client_frac_eval: Rel. share of clients, which to sample for testing a model (float; 0.0 - 1.0; optional)
    :param eval_fn: Custom model evaluation function (function; optional)
    :param model: ML model (object)
    :param server_learning_rate: lr for FedAvgM strategy (only required for said strategy)
    :param server_momentum: Momentum for FedAvgM strategy
    :param eta: eta for FedAdam strategy (float; optional)
    :param eta_l: eta for FedAdam strategy (float; optional)
    :param beta_1: beta 1 for FedAdam strategy (float; optional)
    :param beta_2: beta 2 for FedAdam strategy (float; optional)
    :param tau: tau for FedAdam strategy (float; optional)
    :return: Flower Federated Learning strategy (object)
    """

    strategies = ["FedAvg", "FedSGD", "FedAvgM", "FedAdam"]
    assert strategy in strategies, f"Please select a valid FL strategy. Valid choices are {strategies}"

    if strategy == "FedAvgM":
        assert server_learning_rate is not None and server_momentum is not None, \
            "You chose FedAvgM. Make sure to provide learning rate and momentum to the server."

    if (eval_fn is not None and model is None) or (eval_fn is None and model is not None):
        raise AssertionError("Please provide a ML model along with the eval_fn you are passing to the FL server.")

    if eval_fn is not None:
        eval_fn = eval_fn(model)

    client_frac_eval = client_frac_eval if client_frac_eval is not None else client_frac

    if strategy == "FedAvg":
        fl_strategy = fl.server.strategy.FedAvg(fraction_fit=client_frac, fraction_eval=client_frac_eval,
                                                min_available_clients=min_available_clients, eval_fn=eval_fn,
                                                *args, **kwargs)
    elif strategy == "FedSGD":
        fl_strategy = fl.server.strategy.FedSGD(fraction_fit=client_frac, fraction_eval=client_frac_eval,
                                                min_available_clients=min_available_clients, eval_fn=eval_fn,
                                                *args, **kwargs)
    elif strategy == "FedAvgM":
        fl_strategy = fl.server.strategy.FedAvgM(fraction_fit=client_frac, fraction_eval=client_frac_eval,
                                                 min_available_clients=min_available_clients, eval_fn=eval_fn,
                                                 server_learning_rate=0.0001, server_momentum=0.00001,
                                                 *args, **kwargs)
    elif strategy == "FedAdam":
        fl_strategy = fl.server.strategy.FedAdam(fraction_fit=client_frac, fraction_eval=client_frac_eval,
                                                 min_available_clients=min_available_clients, eval_fn=eval_fn,
                                                 eta=eta, eta_l=eta_l, beta_1=beta_1, beta_2=beta_2, tau=tau,
                                                 *args, **kwargs)
    else:
        raise NotImplementedError("Please select a valid FL strategy. Options: FedAvg, FedSGD, FedAvgM, FedAdam")

    return fl_strategy


