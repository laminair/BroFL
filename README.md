# FLEdge - Benchmarking Federated Machine Learning Applications in Edge Computing Systems
FLEdge provides a modular toolkit for quickly building and evaluating federated learning applications in edge computing 
environments.

## Resources
TBA

## Reference implementations
FLEdge provides three reference implementations from different fields of application.

| **#** | **Use case** | **Dataset**                                                                                 | **Model architecture**      | **Source**                                                                                                            | **Pipeline**     | **ML Models**                        |
|-------|--------------|---------------------------------------------------------------------------------------------|-----------------------------|-----------------------------------------------------------------------------------------------------------------------|------------------|--------------------------------------|
| 1     | NILM         | [BLOND](https://www.nature.com/articles/sdata201848)                                        | CNN, LSTM, ResNet, DenseNet | [Schwermer et al.](https://dl.acm.org/doi/abs/10.1145/3538637.3538845)                                                | `blond`          | `cnn`, `lstm`, `resnet`, `densenet`  |
| 2     | CV           | [FEMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)                    | CNN                         | [Cohen et al.](https://arxiv.org/abs/1702.05373v1)                                                                    | `femnist`        | `cnn`                                |
| 3     | NLP          | [Shakespeare Next Character Prediction](https://www.gutenberg.org/cache/epub/100/pg100.txt) | LSTM                        | [LEAF Benchmark](https://leaf.cmu.edu/)                                                                               | `shakespeare`    | `lstm`                               |

**Note:** In the actual system the pipelines are .git submodules. For the submission, we included the three reference implementations directly.

## Available Federated Optimizers
FLench provides 6 plug'n'play federated machine learning optimizers. They work with every pipeline. The major configuration is done on server side.

| **#** | **Optimizer**                | **Field of application / Intended use**                       | **Source**                                          | **Setting**  | **Comment**                                      |
|-------|------------------------------|---------------------------------------------------------------|-----------------------------------------------------|--------------|--------------------------------------------------|
| 1     | Federated Averaging          | Widely used general purpose FL Optimizer                      | [McMahan et al.](https://arxiv.org/abs/1602.05629)  | `fedavg`     |                                                  |
| 2     | Federaged Averaging (q-Fair) | FedAvg focused on increasing distribution uniformity          | [Li et al.](https://openreview.net/pdf?id=ByexElSYDr) | `qfedavg`    |                                                  |
| 3     | Federated Adam               |                                                               | [Reddi et al.](https://arxiv.org/abs/2003.00295)    | `fedadam`    |                                                  |
| 4     | Federated Adaptive Gradients |                                                               | [Reddi et al.](https://arxiv.org/abs/2003.00295)    | `fedadagrad` |                                                  |
| 5     | Federated Yogi               |                                                               | [Reddi et al.](https://arxiv.org/abs/2003.00295)    | `fedyogi`    |                                                  |
| 6     | FedProx                      | Strategy accounting for the amount of work a client has done. | [Li et al.](https://arxiv.org/pdf/1812.06127.pdf)   | `fedprox`    |                                                  |

## Other features

### Energy monitoring
We include thorough energy monitoring along with our hardware monitoring for embedded devices in our system.
The monitor scrapes energy statistics in 1 second intervals along with all other metrics. As ARM devices do not have an
onboard power meter (as bare-metal intel chips) we rely on PoE infrastructure for capturing power readings. The 
monitoring script connects to the DB we use for storing the statistics. If you want to replicate our results, you may 
get any PoE+ switch that offers a Cisco SSH interface. You may use `netmiko` to work with the switch SSH.

### Client Dropouts
We allow client dropout emulation. In FL systems it does not play a role whether clients experience process errors 
or a network outage. In both cases a client does not send a model update. We emulate this behavior with the option of 
defining the number of client dropouts: `--client-dropouts [0.0 - n]` on the FL server. The sever will randomly send 
dropout triggers to clients. Clients subsequently do not train in a given FL round and do not submit an update to the
server. For this feature to work, your FL strategy nees to support `accept_failure` (see flower docs for details).
**Important note**: A client dropout actually kills the training process and shuts down a client. You need to restart
the client on failure to re-register on the server. This can be done with systemctl, for example.

### Differential Privacy
We are support Opacus-based differential privacy. You may enable it `--use-dp True` in the server configuration prompt.
The server then forces the client to run all workloads differentially private. 

#### Parameters
We apply (ε, δ)-DP. ε is the privacy budget, i.e. the maximum distance of the same result on two clients. δ is the 
likelihood data leakage. The lower ε, δ are, the lower the training performance is.
While having no significantly lower compute time per step, we require substantially more training steps to achieve the 
same results as training without DP. Concluding, the convergence rate is slower with DP.

With our DP wrapper for user-level DP, we support client-side and server-side update noising. Further, it is FL strategy
agnostic.

#### Limitations 
We use Opacus to run differentially private workloads. If you wish to implement sample-level DP please note the 
following: As LSTMs a prone to leak data, opacus provides a DPLSTM implementation. So, if you are using LSTMs in your 
model make sure to replace it. Also note that some normalization/regularization layers are not DP conform. As such, you 
must replace them. See here for details: https://github.com/pytorch/opacus/blob/main/Migration_Guide.md#privacy-accounting


## Platform dependencies
Installation of FLEdge requires the following prerequisites: 

```
Platform requirements: 
    Operating System:       Ubuntu 18.04 / 20.04+ 
    Python:                 Python 3.6 / 3.9
    libsndfile1:            1.0.28-7ubuntu0.1
    unzip:                  6.00
    
Python packages
    PyTorch:                1.10.1 / 1.13.1
    PyTorch Lightning:      1.5.10 / 1.9.4
    Flower:                 0.18.0 (used for the paper) / 1.3.0
    and respective dependencies that are compatible with above mentioned packages (Installable via requirements.txt).
```
**Important note**: The implementation for the FLEdge paper is available in branch "fledge-paper". It contains an 
older version of our code that is compatible with Ubuntu 18.04 LTS and Python 3.6. This is due to the recent deprecation
Nvidia Jetson Nano (2020) platform.

## Installation
We provide three native installation methods: 
- Ansible for bare-metal deployments
- Terraform & Ansible for Cloud deployments
- Docker for platform-agnostic deployment

The automation is available in the project's `automation` folder.

### Ansible for bare-metal deployments
To run the installation process with Ansible only, you need to create an inventory `ansible/inventory_static.cfg` first.
It must contain a server instance and as many clients as you like. Below is an example of an ansible inventory.
In case you want the experiment automation to work, you must label the server starting with `[fledge_server]` and the 
clients starting with `[fledge_client]`. You may put any character combination after these labels.

```editorconfig
[fledge_client_<YOUR IDENTIFIER>]
192.168.0.2
192.168.0.3
192.168.0.4
...

[fledge_server]
192.168.1.1
...
```

### Terraform & Ansible for cloud deployments
To quickly ramp up experiments or a testbench. the cloud is the best place to go. We provide a convenient integration of
Terraform and Ansible. For this to run you need to configure Terraform first and provide the correct `provider` and 
instance configurations. Below are examples: 

**fledge.tf**
```terraform
terraform {
  required_providers {
    opennebula = {
      source  = "opennebula/opennebula" # Set this to your specific cloud provider, e.g., AWS or Azure
      version = "1.1.1"
    }
  }
}

provider "opennebula" {
  username      = var.ON_USERNAME
  password      = var.ON_PASSWD
  endpoint      = "http://xml-rpc.xxx.com"
  flow_endpoint = "http://oneflow.xxx.com"
  insecure      = true
}

...
```

Also, in the fledge.tf configuration, you must provide instance configurations. For this, take a look at the official
terraform docs for your specific cloud provider. Here's a starting point: [Terraform Docs on Providers](https://registry.terraform.io/browse/providers).

**variables.tf**
You'll also find a variables file in the terraform automation directory. Make sure to include all relevant variables 
there. Below is an example of variables required for OpenNebula: 

```terraform
variable "ON_USERNAME" {
  description = "OpenNebula Username"
  type        = string
  default     = ""
}

variable "ON_PASSWD" {
  description = "OpenNebula Password"
  type        = string
  default     = ""
  sensitive   = true
}
```

**credentials.tfvars**
This file must never be included in a commit! Use this file to store your cloud provider credentials. It may look like 
so:

```terraform
ON_USERNAME = "USERNAME"
ON_PASSWD = "YOUR PASSWORD"
```

Now that this is all set up, we're good to go. Initialize Terraform from inside the terraform directory first. 
Then, use `automation/deploy.sh` to run the Terraform & Ansible automation. 
The Terraform script automatically creates an ansible inventory for you and applies the node configuration after 45 
seconds.

### Teardown a project deployed with Terraform and Ansible
We provide a script to undeploy the project. It removed virtualized resources and removes the project (incl. python 
dependencies) from bare-metal machines. It does not remove APT system resources as this may break your system. The 
script is available under `automation/teardown.sh` 

### Docker automation
We provide a set of minimal docker images that you may use for your experiments (Pull available from [DockerHub](https://hub.docker.com/repositories/her3ert)). Please note you need to include `sampled data` 
and the `.netrc` file yourself. The Dockerfiles for building the images below are available in the `docker` folder.

| Image name      | Version | Platform      | Python | PyTorch | PT Lightning | Flower | cuDNN | Details                      |
|-----------------|---------|---------------|--------|---------|--------------|--------|-------|------------------------------|
| NVIDIA GPU      | 2.0.0   | Ubuntu 20.04  | 3.9    | 1.13    | 1.9.4        | 1.3.0  | 8.2   | Generic NVIDIA GPU support   |
| Generic         | 2.0.0   | Ubuntu 20.04  | 3.9    | 1.13    | 1.9.4        | 1.3.0  | N/A   | CPU-only; no HW-acceleration |
|                 |         |               |        |         |              |        |       |                              |
| Jetson Nano 2GB | 1.0.0   | Ubuntu 18.04  | 3.6    | 1.10    | 1.5.10       | 0.18.0 | 7.x   | GPU support for Jetson Nano  |
| NVIDIA GPU      | 1.0.0   | Ubuntu 18.04  | 3.6    | 1.10    | 1.5.10       | 0.18.0 | 8.2   | Generic NVIDIA GPU support   |
| Generic         | 1.0.0   | Ubuntu 18.04  | 3.6    | 1.10    | 1.5.10       | 0.18.0 | N/A   | CPU-only; no HW-acceleration |

The minimal images do not contain any datasets. Please close the repository, download the datasets and sample the data 
as required for your purposes. You may then build you own docker image with datasets included. The commands should be 
run from the project root directory. To build the latest images: 

#### AMD64 Generic
```commandline
docker build -t <YOUR_NAME>/fledge:2.2.0-amd64 -f ./docker/v2.2.0/Dockerfile_generic_amd64 .
```

#### AMD64 GPU-accelerated
```commandline
docker build -t <YOUR_NAME>/fledge:2.2.0-amd64-nvidia -f ./docker/v2.2.0/Dockerfile_nvidia_amd64 .
```
#### ARM64 Generic
This image can also be run on Jetson Nano 2GB (without GPU acceleration)
```commandline
docker build -t <YOUR_NAME>/fledge:2.2.0-arm64 -f ./docker/v2.2.0/Dockerfile_generic_arm64 .
```

#### Nvidia Jetson Nano 2GB
Please note that the last supported OS was 18.04 LTS and PyTorch 1.10 with CUDA 10.3. There will be no further updates
on this image beyond v1.0.0
```commandline
docker build -t <YOUR_NAME>/fledge:1.0.0-arm64-jnano -f ./docker/v1.0.0/Dockerfile_jnano_arm64 .
```

## Including your own pipelines
We designed FLEdge such that extending it with additional ML pipelines becomes simple.
For this we require three things: PyTorch Lightning as a basis, a PyTorch Lightning Module for our ML model, and a 
PyTorch Lightning Data Module for data loading. The configuration for your pipeline must be included in `config.json`.
You may include preprocessing as needed. Our built-in pipelines provide examples on how to preprocess data.

### Lightning Module
The PyTorch Lightning module must have the following signature to work with FLEdge. A full reference is available in 
the [PyTorch Lightning Docs](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html).

```python
import pytorch_lightning as pl

class YourLightningModule(pl.LightningModule):

    def __init__(self, <"your args go here and must correspond to what you provide in config.json">, *args, **kwargs):
        ...
```

### Lightning Data Module
This is where data loading happens. You may use `config.json` to set the data loading hyperparameters correctly. 
A full reference is available in the [PyTorch Lightning Docs](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html).
The signature should look similar to: 

```python
import pytorch_lightning as pl

class YourLightningDataModule(pl.LightningModule):

    def __init__(self, stage="train", transform=None, data_dist="local", client_id=0, *args, **kwargs):
        ...
```

## Using FLEdge
Our design allows to run local and federated experiments with an arbitrary number of clients and a single server endpoint.

![FLEdge System Design](evaluations/paper_figures_OLD/experiment_design/220927_FL_System_Conceptual.png)

### Running baseline experiments locally on a single device
Running a baseline provides the best possible results of an ML pipeline and lays the ground for doing a sensitivity 
analysis of your ML pipeline with different FL optimizers and data distributions. Our examples provide you with an 
implementation of creating subsets using a Dirichlet distribution. 

## Running workloads
Workloads are ML pipelines attached to our client-server architecture. We leave it up to you what optimizer you want to 
use. Check `config.json` for the FL optimizer parameterization we chose or see the official [Flower Docs](https://flower.dev/docs/) how to reparameterize.

**Important note**: By default, the system is running with `scapy` disabled. It allows to monitor network traffic between
clients and server. To use scapy, you need to run the server and device with sudo rights and set the flag 
`use_scapy_network_monitoring` in `config.json`. With scapy disabled, sudo is not required. 

### Launching a baseline experiment
All registered ML pipelines can be run locally for retrieving baseline performance metrics without effects of FL 
optimizers. You need to initiate a client with a pipeline, a single machine, local data distribution, and the number of 
training epochs. Here's how you launch it:
```commandline
venv/bin/python3 device.py --client-id 0 --pipeline blond --ml-model cnn --data-dist local --experiment-name test
```

### Remarks about strategies
We changed the all Flower strategies to take care of assigning clients their IDs in the `configure_fit` and 
`configure_evaluate` functions of the FedAvg and qFedAvg strategies. Therefore, we import all strategies from inside the FLEdge 
project. This does not change the functional behavior. You only get more configuration options.

**Important note:** FLEdge is now using derivatives of Flower's built-in strategies to allow for greater flexibility. 
You can find them in `modules/strategies`. The functionality in terms of how models are merged has not changed. With the
derivative implementations, we are able to configure clients individually via the server and do not have to handle extensive
SSH instructions.

### Launching federated experiments
The purpose of FLEdge is to explor FL workloads in embedded devices and in edge computing systems. We provide a `server`
and `device` endpoint to configure the FL system.

**Launching the server** (minimal parameterization)

The server is data distribution agnostic. However, we recommend you pass the data distribution as part of the experiment 
name.
```commandline
python3 server.py --pipeline blond --ml-model cnn --data-dist iid --fl-strategy fedavg --training-rounds 10 --num-clients 2 --client-dropouts 0 --use-dp False  --server-endpoint 127.0.0.1:8080 --experiment-name test
```

**Launching clients**

Clients do need information about the data distribution they should use as part of their workload. Repeat the command 
below as many times as required. The server will send all configuration instructions to the client just before training
or evaluation/testing. This renders clients ephemeral.
```commandline
venv/bin/python3 device.py --server-endpoint 127.0.0.1:8080
venv/bin/python3 device.py --server-endpoint 127.0.0.1:8080
venv/bin/python3 device.py --server-endpoint 127.0.0.1:8080
...
```

## Experiment automation
We provide an experiment automation script based on SSH instructions and tmux sessions.
The current automation to run all pipelines in combination with all data distributions and FL optimizers can be found in 
`automation/experiments`. The script will handle spawning jobs on all clients automatically. However, before you run the
automation, make sure to install all dependencies with Ansible.

Running the automation can be done with the following command: 
```commandline
python3 automation/experiments/fledge.py --inventory-path automation/ansible/inventory.cfg --server-endpoint 127.0.0.1:8080
```

## Simulation 
We implement flower-based simulation as part of the server. Instead of launching a real server with an exposed port, we 
launch a simulation instance. To start a simulation, you may use the command below. It differs from the physical server
only by the configuration parameter `--is-simulation True`. However, it implements the same capabilities as real FL 
training:
```commandline
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 server.py --pipeline blond --ml-model cnn --data-dist iid --fl-strategy fedavg --training-rounds 5 --num-clients 4 --client-dropouts 0 --use-dp False --dp-noise-multiplier 0 --server-endpoint 127.0.0.1:8080 --is-simulation True --experiment-name test
```

## Known issues
Here we list issues related to external libraries and how to circumvent them.

### Scikit-learn throws an error that TLS blocks may not be allocated.
When running scikit-learn on Aarch64, you must preload the `libgomp` library. Refer here for more details: [StackOverflow](https://stackoverflow.com/questions/67735216/after-using-pip-i-get-the-error-scikit-learn-has-not-been-built-correctly)
For FLEdge, this is done like so: 
```commandline
export LD_PRELOAD='/opt/fledge/venv/lib/python3.9/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0'
```
When running FLEdge with scapy monitoring, you must preserve the bash environment with `sudo -E` to maintain the 
exported variable or export the variables o the sudo/root environment.

### Running the TextClassification Task on Apple Silicon with MPS acceleration
When running the text classification task on Apple Silicon Devices, we found an unsupported operation. This requires to
set an environment variable before starting the training `PYTORCH_ENABLE_MPS_FALLBACK=1`. Here's the full command: 
```commandline
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 device.py ...
```

## License notes
FLEdge is released under Apache license v2.0. See `LICENSE` for details.
All libraries used in this project are subject to their own licenses.