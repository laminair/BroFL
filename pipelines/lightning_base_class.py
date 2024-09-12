import torch
import pytorch_lightning as pl
import torchmetrics
import time

from torch.nn import functional as F


class FLEdgeLightningBase(pl.LightningModule):
    """
    Implements the BLOND CNN classifier as PyTorch Lightning module.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Here we store the hyperparameters in a dict.
        self.config = {**vars(*args)}
        self.config.update(kwargs)

        # The mu is used to parameterized the loss when using FedProx
        self.use_fedprox = False
        self.mu = 0
        self.init_params = None

        # Metrics
        self.criterion = None
        self.optim = None

        # if "model_name" not in self.config.keys() or "t5" not in self.config["model_name"]:
        #     self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.config["num_classes"])
        #     self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.config["num_classes"])

        # Timing
        self.batch_load_time_start = time.time()

        # FedProx requirements
        self.init_params = None
        self.mu = None

    def forward(self, x):
        raise NotImplementedError("Please provide a forward step to your model.")

    def set_loss_function(self, loss_criterion=F.cross_entropy):
        """
        This method sets the loss function for our model.
        """
        self.criterion = loss_criterion

    def training_step(self, batch, batch_i):
        assert self.criterion is not None, "Make sure to set the loss function before starting training."
        assert self.optim is not None, "Make sure to set the optimizer before starting training."
        return self._training_step(batch, batch_i)

    def validation_step(self, batch, batch_i):
        return self._step(batch, batch_i, stage="val")

    def test_step(self, batch, batch_i):
        return self._step(batch, batch_i, stage="test")

    def validation_epoch_end(self, outputs):
        super().validation_epoch_end(outputs)

    # def test_epoch_end(self, outputs) -> None:
    #     acc_list = []
    #     loss_list = []
    #     for el in outputs:
    #         try:
    #             acc_list.append(el["accuracy"])
    #         except KeyError:
    #             pass
    #
    #         try:
    #             loss_list.append(el["loss"])
    #         except KeyError:
    #             pass
    #
    #     if len(acc_list) > 0:
    #         acc = torch.mean(torch.stack(acc_list)).item()
    #     else:
    #         acc = 0
    #
    #     if len(loss_list) > 0:
    #         loss = torch.mean(torch.stack(loss_list)).item()
    #     else:
    #         loss = 1000
    #
    #     try:
    #         self.logger.log_metrics({
    #             "test/accuracy": acc,
    #             "test/loss": loss
    #         })
    #     except AttributeError:
    #         pass

    def configure_optimizers(self):
        return self.optim

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int = 0, optimizer_closure=None,
                       on_tpu: bool = False, using_lbfgs: bool = False, *args, **kwargs) -> None:
        start_time = time.time()
        optimizer.step(closure=optimizer_closure)
        self.log("timing/train/optimizer_s", time.time() - start_time)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs) -> None:
        start_time = time.time()
        super().backward(loss, optimizer, optimizer_idx, *args, **kwargs)
        self.log("timing/train/backward_s", time.time() - start_time)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.batch_load_time_start = time.time()

    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx)
        self.log("timing/train/batch_load_time_s", time.time() - self.batch_load_time_start)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        # We reset the timing after a processing step to calculate the batch loading time again.
        self.batch_load_time_start = time.time()

    def set_mu(self, init_parameters, mu):
        """
        This function sets the proximity term when using the FedProx strategy.
        :param mu: Used to define the FedProx proximity term (float).
        :param init_parameters: We need to pass the model parameters a client gets from the server so we can calculate
                                the distance between parameter adjustments during training compared to the initial
                                state.
        """
        self.mu = mu
        self.init_params = init_parameters
        self.use_fedprox = True

    ####################################################################################################################
    #                                                  Support Functions                                               #
    ####################################################################################################################
    def _training_step(self, batch, batch_i):
        x, y = batch

        start_time = time.time()
        y_hat = self.forward(x)
        self.log("timing/train/forward_time", time.time() - start_time)

        start_time = time.time()
        loss = self.calculate_loss(y=y, y_hat=y_hat, use_fedprox=self.use_fedprox, mu=self.mu,
                                   params=self.parameters() if self.use_fedprox else None, init_params=self.init_params)
        self.log("timing/train/loss_calc_time_s", time.time() - start_time)

        _, y_hat = torch.max(y_hat, 1)
        self.log("train/loss", loss)

        res = {"loss": loss, "y_hat": y_hat}

        if "num_classes" in self.config.keys():
            accuracy = self.acc(y_hat, y)
            f1 = self.f1(y_hat, y)
            res.update({"accuracy": accuracy, "f1_score": f1})
            self.log("train/accuracy", accuracy)

        return res["loss"]

    def _step(self, batch, batch_i, stage=None):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        _, y_hat = torch.max(y_hat, 1)

        res = {"loss": loss, "y_hat": y_hat}

        # print(f"STEP: {res}")

        # if "num_classes" in self.config.keys():
        accuracy = self.acc(y_hat, y)
        f1 = self.f1(y_hat, y)
        res.update({"accuracy": accuracy, "f1_score": f1})

        for k, v in res.items():
            if k != "y_hat":
                self.log(f"{stage}/{k}", v)

        return res

    def calculate_loss(self, y, y_hat, use_fedprox: bool = False, mu: float = 0, params=None, init_params=None):
        """
        This function helps us to unify the way we calculate the loss and adapt FedProx in our system.
        This method is only required in the training_step as we do not want to adjust the evaluation loss. This would
        have no effect.
        :param y: ground truth for criterion (tensor)
        :param y_hat: predictions for criterion (tensor)
        :param use_fedprox: Indicates whether we want to use FedProx as strategy (bool)
        :param mu: Parameter for the FedProx proximity term (float)
        :param params: Current model parameters (state_dict)
        :param init_params: Initial model parameters before training started (state_dict)
        """
        loss = self.criterion(y_hat, y)

        if use_fedprox:

            # First we need to calculate the proximal term
            prox_term = 0
            for local_params, global_params in zip(params, init_params):
                prox_term += (local_params.detach() - global_params).norm(2)

            # We now adjust the loss for the proximal term.
            loss += (mu / 2) * prox_term

        return loss
