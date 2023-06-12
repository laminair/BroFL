"""
Implementations of CNN, LSTM, ResNet, DenseNet for the BLOND dataset. Adapted from FederatedBlond by René Schwermer (TUM)
Author: Herbert Woisetschläger, TUM
"""
import pytorch_lightning as pl
import torch
import torchmetrics
import time

from torch import nn
from torch.nn import functional as F


########################################################################################################################
#                                                 BLOND Base Class                                                     #
########################################################################################################################
class BLONDLightningBase(pl.LightningModule):

    def __init__(self, in_features=68, seq_len=190, num_classes=12, out_features=10, num_layers=1, use_npy=True,
                 privacy_engine=None, *args, **kwargs):
        super(BLONDLightningBase, self).__init__()
        self.save_hyperparameters()

        self.config = {**vars(*args)}
        self.config.update({**kwargs})

        self.seq_len_adj = 0 if self.config["use_npy"] is False else 1
        self.classifier = BlondNetMLP(self.config["seq_len"], self.config["in_features"], self.config["num_classes"],
                                      max(1, int(self.config["num_layers"] / 2)), seq_len_adj=self.seq_len_adj)
        self.optim = torch.optim.SGD(lr=0.052, weight_decay=0.001, params=self.parameters())

        # Differential privacy basis
        # if self.config["use_dp"]:
        #     # The privacy engine must be maintained at device level to track epsilon over the entire training process.
        #     self.privacy_engine = privacy_engine
        #     assert self.privacy_engine is not None, "Please make sure to pass an Opacus Privacy Engine to the model."
        #     self.epsilon = None

        # Metrics
        self.criterion = F.cross_entropy
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.config["num_classes"])
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.config["num_classes"])

        # Timing
        self.batch_load_time_start = time.time()

        # The mu is used to parameterized the loss when using FedProx
        self.use_fedprox = False
        self.mu = 0
        self.init_params = None

    def training_step(self, batch, batch_i):
        return _training_step(self, batch, batch_i)

    def validation_step(self, batch, batch_i):
        return _step(self, batch, batch_i, stage="val")

    def test_step(self, batch, batch_i):
        return _step(self, batch, batch_i, stage="test")

    def validation_epoch_end(self, outputs):
        super(BLONDLightningBase, self).validation_epoch_end(outputs)

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
        super(BLONDLightningBase, self).backward(loss, optimizer, optimizer_idx, *args, **kwargs)
        self.log("timing/train/backward_s", time.time() - start_time)

    def on_train_epoch_start(self) -> None:
        super(BLONDLightningBase, self).on_train_epoch_start()
        self.batch_load_time_start = time.time()

    def on_train_batch_start(self, batch, batch_idx):
        super(BLONDLightningBase, self).on_train_batch_start(batch, batch_idx)
        self.log("timing/train/batch_load_time_s", time.time() - self.batch_load_time_start)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super(BLONDLightningBase, self).on_train_batch_end(outputs, batch, batch_idx)
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


########################################################################################################################
#                                                    CNN Estimator                                                     #
########################################################################################################################
class BlondLightningCNN(pl.LightningModule):
    """
    Implements the BLOND CNN classifier as PyTorch Lightning module.
    """
    def __init__(self, in_features=68, seq_len=190, num_classes=12, out_features=10, num_layers=1, use_npy=True,
                 *args, **kwargs):
        super(BlondLightningCNN, self).__init__()
        self.save_hyperparameters()
        self.use_npy = use_npy

        # The mu is used to parameterized the loss when using FedProx
        self.use_fedprox = False
        self.mu = 0
        self.init_params = None

        seq_len_adj = 0 if use_npy is False else 1

        self.layers = nn.ModuleList()
        for i in range(0, num_layers):
            layer = BlondConvNetLayer(in_features, seq_len, out_features)
            self.layers.append(layer)

            # Assign values for next layer
            seq_len = layer.seq_len
            in_features = out_features
            out_features = int(out_features * 1.5)

        self.classifier = BlondNetMLP(seq_len, in_features, num_classes, max(1, int(num_layers / 2)), seq_len_adj=seq_len_adj)
        self.optim = torch.optim.SGD(lr=0.055, weight_decay=0, params=self.parameters())

        # Metrics
        self.criterion = F.cross_entropy
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

        # Timing
        self.batch_load_time_start = time.time()

        # FedProx requirements
        self.init_params = None
        self.mu = None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_i):
        return _training_step(self, batch, batch_i)

    def validation_step(self, batch, batch_i):
        return _step(self, batch, batch_i, stage="val")

    def test_step(self, batch, batch_i):
        return _step(self, batch, batch_i, stage="test")

    def validation_epoch_end(self, outputs):
        super(BlondLightningCNN, self).validation_epoch_end(outputs)

    def test_epoch_end(self, outputs) -> None:
        try:
            self.logger.log_metrics({
                "test/accuracy": torch.mean(torch.stack([el["accuracy"] for el in outputs])).item(),
                "test/loss": torch.mean(torch.stack([el["loss"] for el in outputs])).item()
            })
        except AttributeError:
            pass

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


class BlondConvNetLayer(nn.Module):

    def __init__(self, in_features, seq_len, out_features):
        """
        Args:
            in_features (int): Number of input features
            seq_len (int): Length of input series
            out_features (int): Size of hidden layer
        """
        super(BlondConvNetLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_features, track_running_stats=False),  # We swap out BatchNorm1d for GroupNorm to be DP conform
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.ReLU())

        self.seq_len = self._calc_dims(self.layer, seq_len)

    def _calc_dims(self, layer, seq_len):
        """
        Args:
            layer (nn.Sequential): Current layer
            seq_len (int): Length of input series

        Returns:
            (int): Series length of the layer output
        """
        seq_len = int((seq_len + (2 * layer[0].padding[0]) - layer[0].dilation[0] * (layer[0].kernel_size[0] - 1) - 1) /
                      layer[0].stride[0] + 1)

        seq_len = int((seq_len + (2 * layer[2].padding) - layer[2].dilation * (layer[2].kernel_size - 1) - 1) / layer[
            2].stride + 1)

        return seq_len

    def forward(self, x):
        x = self.layer(x)

        return x


########################################################################################################################
#                                                   LSTM Estimator                                                     #
########################################################################################################################
class BlondLightningLSTM(BLONDLightningBase):

    def __init__(self, in_features=68, seq_len=190, num_classes=12, hidden_layer_size=15, num_layers=1, use_npy=True,
                 privacy_engine=None, *args, **kwargs):
        super(BlondLightningLSTM, self).__init__(in_features=in_features, seq_len=seq_len, num_classes=num_classes,
                                                 hidden_layer_size=hidden_layer_size, num_layers=num_layers,
                                                 use_npy=use_npy, privacy_engine=privacy_engine, *args, **kwargs)

        self.seq_len_adj = 0 if self.config["use_npy"] is False else 2

        self.num_layers = self.config["num_layers"]
        self.hidden_layer_size = self.config["hidden_layer_size"]

        # if self.config["use_dp"]:
        #     self.lstm = DPLSTM(self.config["in_features"], self.config["hidden_layer_size"], dropout=0,
        #                        num_layers=self.config["num_layers"], batch_first=True)
        # else:
        self.lstm = nn.LSTM(self.config["in_features"], self.config["hidden_layer_size"], dropout=0,
                            num_layers=self.config["num_layers"], batch_first=True)
        self.classifier = BlondNetMLP(self.config["seq_len"], self.config["hidden_layer_size"],
                                      self.config["num_classes"], max(1, int(self.config["num_layers"] / 2)),
                                      seq_len_adj=self.seq_len_adj
                                      )

        self.optim = torch.optim.SGD(lr=0.045, weight_decay=0.001, params=self.parameters())
        self.criterion = F.cross_entropy

    def forward(self, x):

        h0 = torch.zeros(self.config["num_layers"], x.size(0),
                         self.config["hidden_layer_size"]).requires_grad_().to(self.device)
        c0 = torch.zeros(self.config["num_layers"], x.size(0),
                         self.config["hidden_layer_size"]).requires_grad_().to(self.device)

        x = x.transpose(2, 1)
        x, _ = self.lstm(x, (h0.detach(), c0.detach()))
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x


########################################################################################################################
#                                                  ResNet Estimator                                                    #
########################################################################################################################
class BlondLightningResNet(BLONDLightningBase):

    def __init__(self, in_features=68, num_classes=12, out_features=28, num_layers=4, use_npy=True, privacy_engine=None,
                 *args, **kwargs):
        super(BlondLightningResNet, self).__init__(in_features=in_features, num_classes=num_classes,
                                                   out_features=out_features, num_layers=num_layers,
                                                   use_npy=use_npy, privacy_engine=privacy_engine, *args, **kwargs)

        in_features = self.config["in_features"]
        out_features = self.config["out_features"]
        self.layers = nn.ModuleList()

        for i in range(0, self.config["num_layers"]):
            layer = BlondResNetLayer(in_features, out_features)
            self.layers.append(layer)

            # Assign values for next layer
            in_features = out_features
            out_features = int(out_features * 1.5)

        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.classifier = BlondNetMLP(1, in_features, self.config["num_classes"], max(1, int(self.config["num_layers"] / 2)))
        self.optim = torch.optim.SGD(lr=0.052, weight_decay=0.001, params=self.parameters())

        self.criterion = F.cross_entropy
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.config["num_classes"])
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.config["num_classes"])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.global_avg(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x


class BlondResNetLayer(nn.Module):

    def __init__(self, in_features, out_features):
        """
        Args:
            in_features (int): Number of input features
            out_features (int): Size of hidden layer
        """
        super(BlondResNetLayer, self).__init__()

        self.skip_connection = None
        if in_features != out_features:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_features, track_running_stats=False),
            )  # Swap BatchNorm1d for GroupNorm

        self.layer = nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_features, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(out_features, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_features, track_running_stats=False),
        )  # Swap BatchNorm1d for GroupNorm

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.layer(x)
        if self.skip_connection:
            identity = self.skip_connection(identity)
        x += identity
        x = self.relu(x)

        return x


########################################################################################################################
#                                                 DenseNet Estimator                                                   #
########################################################################################################################
class BlondLightningDenseNet(BLONDLightningBase):

    def __init__(self, in_features=68, num_classes=12, out_features=32, num_layers=4, use_npy=True, privacy_engine=None,
                 *args, **kwargs):
        super(BlondLightningDenseNet, self).__init__(in_features=in_features, num_classes=num_classes,
                                                     out_features=out_features, num_layers=num_layers,
                                                     use_npy=use_npy, privacy_engine=privacy_engine, *args, **kwargs)

        in_features = self.config["in_features"]
        out_features = self.config["out_features"]
        self.layers = nn.ModuleList()
        for i in range(0, self.config["num_layers"]):
            layer = BlondDenseLayer(in_features, out_features)
            self.layers.append(layer)

            if i < self.config["num_layers"] - 1:
                transition = BlondDenseTransitionLayer(in_features + 2 * out_features, 64)
                self.layers.append(transition)

                in_features = 64
                out_features = int(out_features * 1.5)

        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.classifier = BlondNetMLP(1, in_features + 2 * out_features, self.config["num_classes"],
                                      max(1, int(self.config["num_layers"] / 2)))
        self.optim = torch.optim.SGD(lr=0.075, weight_decay=0.001, params=self.parameters())

        self.criterion = F.cross_entropy
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.config["num_classes"])
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.config["num_classes"])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.global_avg(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x


class BlondDenseLayer(nn.Module):
    """
    Args:
        in_features (int): Number of input features
        out_features (int): Size of out_channels of convolutional block
    """

    def __init__(self, in_features, out_features):
        super(BlondDenseLayer, self).__init__()

        self.layer1 = nn.Sequential(
            nn.BatchNorm1d(in_features, track_running_stats=False),  # Swap BatchNorm1d for GroupNorm
            nn.ReLU(),
            nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_features, track_running_stats=False),  # Swap BatchNorm1d for GroupNorm
            nn.ReLU(),
            nn.Conv1d(out_features, out_features, kernel_size=3, stride=1, padding=1)
        )

        in_features = in_features + out_features
        self.layer2 = nn.Sequential(
            nn.BatchNorm1d(in_features, track_running_stats=False),  # Swap BatchNorm1d for GroupNorm
            nn.ReLU(),
            nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_features, track_running_stats=False),  # Swap BatchNorm1d for GroupNorm
            nn.ReLU(),
            nn.Conv1d(out_features, out_features, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        lay1 = self.layer1(x)
        x1 = torch.cat([x, lay1], 1)
        lay2 = self.layer2(x1)
        out = torch.cat([x, lay1, lay2], 1)
        return out


class BlondDenseTransitionLayer(nn.Module):
    """
   Args:
       in_features (int): Number of input features
       out_features (int): Size of out_channels of convolutional block
   """
    def __init__(self, in_features, out_features):

        super(BlondDenseTransitionLayer, self).__init__()

        self.transition = nn.Sequential(
            nn.BatchNorm1d(in_features, track_running_stats=False),
            nn.ReLU(),
            nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.transition(x)
        return x


########################################################################################################################
#                                                Classification Layer                                                  #
########################################################################################################################
class BlondNetMLP(nn.Module):
    """
    Final classification layer
    """
    def __init__(self, seq_len, in_features, num_classes, num_layers, seq_len_adj=0):
        super(BlondNetMLP, self).__init__()

        in_size = int(in_features * (seq_len + seq_len_adj))
        self.mlp = nn.Sequential()
        i = 0
        for i in range(1, num_layers):
            self.mlp.add_module(f'Linear({i - 1})', nn.Linear(in_size, int(in_size / 2)))
            in_size = int(in_size / 2)

        self.mlp.add_module(f'Linear({i})', nn.Linear(in_size, num_classes))

    def forward(self, x):
        x = self.mlp(x)

        return x


########################################################################################################################
#                                                  Support Functions                                                   #
########################################################################################################################
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
            prox_term += (local_params.cpu().detach() - global_params).norm(2)

        # We now adjust the loss for the proximal term.
        loss += (mu / 2) * prox_term

    return loss
