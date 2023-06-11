import time
import torch
from torch import nn
from torchmetrics import F1Score, Accuracy
import pytorch_lightning as pl
import datetime


class MNISTCNN(pl.LightningModule):

    def __init__(self, num_classes=62, privacy_engine=None, *args, **kwargs):
        super(MNISTCNN, self).__init__()
        self.save_hyperparameters()

        self.config = {**vars(*args)}
        self.config.update({**kwargs})

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),  # Format: 28 x 28 x batch_size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=12*4*4, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=62)
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

        # Differential privacy basis
        # if self.config["use_dp"]:
        #     # The privacy engine must be maintained at device level to track epsilon over the entire training process.
        #     self.privacy_engine = privacy_engine
        #     assert self.privacy_engine is not None, "Please make sure to pass an Opacus Privacy Engine to the model."
        #     self.epsilon = None

        # Metrics
        self.acc = Accuracy(task="multiclass", num_classes=self.config["num_classes"])
        self.f1 = F1Score(task="multiclass", num_classes=self.config["num_classes"])

        # Timing
        self.batch_load_time_start = time.time()

        # The mu is used to parameterized the loss when using FedProx
        self.use_fedprox = False
        self.mu = 0
        self.init_params = None

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1, 12*4*4)
        x = self.linear(x)

        return x

    def training_step(self, batch, batch_i):
        return _training_step(self, batch, batch_i)

    def validation_step(self, batch, batch_i):
        return _step(self, batch, batch_i, stage="val")

    def test_step(self, batch, batch_i):
        return _step(self, batch, batch_i, stage="test")

    def validation_epoch_end(self, outputs):
        super(MNISTCNN, self).validation_epoch_end(outputs)

    def test_epoch_end(self, outputs) -> None:
        try:
            self.logger.log_metrics({
                "test/accuracy": torch.mean(torch.stack([el["accuracy"] for el in outputs])).item(),
                "test/loss": torch.mean(torch.stack([el["loss"] for el in outputs])).item()
            })
        except AttributeError:
            pass

    def configure_optimizers(self):

        optimizer = self.optim

        # if self.config["use_dp"]:
        #     data_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
        #
        #     if hasattr(self, "dp"):
        #         self.dp["model"].remove_hooks()
        #
        #     dp_model, optimizer, dataloader = self.privacy_engine.make_private(
        #         noise_multiplier=self.config["noise_multiplier"],
        #         max_grad_norm=self.config["max_grad_norm"],
        #         module=self,
        #         optimizer=self.optim,
        #         data_loader=data_loader,
        #         poisson_sampling=isinstance(data_loader, DPDataLoader)
        #     )
        #     self.dp = {"model": dp_model}

        return optimizer

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int = 0, optimizer_closure=None,
                       on_tpu: bool = False, using_lbfgs: bool = False, *args, **kwargs) -> None:
        start_time = time.time()
        optimizer.step(closure=optimizer_closure)
        self.log("timing/train/optimizer_s", time.time() - start_time)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs) -> None:
        start_time = time.time()
        super().backward(loss, optimizer, optimizer_idx, *args, **kwargs)
        duration = time.time() - start_time
        self.log("timing/train/backward_s", duration)

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

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        self.batch_transfer_start_time = time.time()
        super().on_before_batch_transfer(batch, dataloader_idx)
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        super().on_after_batch_transfer(batch, dataloader_idx)
        try:
            self.logger.experiment.log("timing/train/batch_transfer_s", time.time() - self.batch_transfer_start_time)
        except AttributeError:
            pass

        return batch

    # def on_train_epoch_end(self):
    #     if self.config["use_dp"]:
    #         # We need this to evaluate the privacy budget spent.
    #         eps = self.privacy_engine.get_epsilon(delta=self.config["target_delta"])
    #         self.log("training/privacy/target_epsilon", eps)
    #         self.log("training/privacy/true_epsilon", eps)
    #         self.epsilon = eps
    #         print("Epsilon: ", eps)

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


def _training_step(self, batch, batch_i):
    x, y = batch

    start_time = time.time()
    y_hat = self.forward(x)
    self.log("timing/train/forward_time", time.time() - start_time)

    start_time = time.time()
    loss = calculate_loss(criterion=self.criterion, y=y, y_hat=y_hat, use_fedprox=self.use_fedprox, mu=self.mu,
                          params=self.parameters() if self.use_fedprox else None, init_params=self.init_params)
    self.log("timing/train/loss_calc_time_s", time.time() - start_time)

    _, y_hat = torch.max(y_hat, 1)
    accuracy = self.acc(y_hat, y)

    self.log("train/loss", loss)
    self.log("train/accuracy", accuracy)
    return loss


def _step(self, batch, batch_i, stage=None):
    x, y = batch
    y_hat = self.forward(x)
    loss = self.criterion(y_hat, y)
    _, y_hat = torch.max(y_hat, 1)
    accuracy = self.acc(y_hat, y)
    f1 = self.f1(y_hat, y)

    res = {"loss": loss, "y_hat": y_hat, "accuracy": accuracy, "f1_score": f1}

    for k, v in res.items():
        if k != "y_hat":
            self.log(f"{stage}/{k}", v)

    return res


def calculate_loss(criterion, y, y_hat, use_fedprox: bool = False, mu: float = 0, params=None, init_params=None):
    """
    This function helps us to unify the way we calculate the loss and adapt FedProx in our system.
    This method is only required in the training_step as we do not want to adjust the evaluation loss. This would have
    no effect.
    :param criterion: Loss function (object)
    :param y: ground truth for criterion (tensor)
    :param y_hat: predictions for criterion (tensor)
    :param use_fedprox: Indicates whether we want to use FedProx as strategy (bool)
    :param mu: Parameter for the FedProx proximity term (float)
    :param params: Current model parameters (state_dict)
    :param init_params: Initial model parameters before training started (state_dict)
    """
    loss = criterion(y_hat, y)

    if use_fedprox:

        # First we need to calculate the proximal term
        prox_term = 0
        for local_params, global_params in zip(params, init_params):
            prox_term += (local_params.detach() - global_params).norm(2)

        # We now adjust the loss for the proximal term.
        loss += (mu / 2) * prox_term

    return loss

