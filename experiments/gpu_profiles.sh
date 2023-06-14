#!/bin/bash

# This script executes the GPU profiler for all workloads. We use nsight systems.
echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
sudo -E PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python nsys profile /opt/flbench/venv38/bin/python3 device.py --client-id 0 --pipeline samsum --ml-model t5 --data-dist local --experiment-name 2023-05-31_17:38_flbench_experiment-plan-baseline_samsum_t5_None_local_1_rounds_1_clients_0_dropout_nodp_0_prec_16
sleep 30
echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
sudo -E PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python nsys profile /opt/flbench/venv38/bin/python3 device.py --client-id 0 --pipeline mnist --ml-model cnn --data-dist local --experiment-name 2023-05-31_17:38_flbench_experiment-plan-baseline_mnist_cnn_None_local_1_rounds_1_clients_0_dropout_nodp_0_prec_16
sleep 30
echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
sudo -E PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python nsys profile /opt/flbench/venv38/bin/python3 device.py --client-id 0 --pipeline mnist --ml-model cnn --data-dist local --experiment-name 2023-05-31_17:38_flbench_experiment-plan-baseline_mnist_cnn_None_local_1_rounds_1_clients_0_dropout_nodp_0_prec_16
sleep 30
echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
sudo -E PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python nsys profile /opt/flbench/venv38/bin/python3 device.py --client-id 0 --pipeline blond --ml-model cnn --data-dist local --experiment-name 2023-05-31_17:38_flbench_experiment-plan-baseline_blond_cnn_None_local_1_rounds_1_clients_0_dropout_nodp_0_prec_16
sleep 30
echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
sudo -E PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python nsys profile /opt/flbench/venv38/bin/python3 device.py --client-id 0 --pipeline blond --ml-model lstm --data-dist local --experiment-name 2023-05-31_17:38_flbench_experiment-plan-baseline_blond_lstm_None_local_1_rounds_1_clients_0_dropout_nodp_0_prec_16
sleep 30
echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
sudo -E PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python nsys profile /opt/flbench/venv38/bin/python3 device.py --client-id 0 --pipeline blond --ml-model resnet --data-dist local --experiment-name 2023-05-31_17:38_flbench_experiment-plan-baseline_blond_resnet_None_local_1_rounds_1_clients_0_dropout_nodp_0_prec_16
sleep 30
echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
sudo -E PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python nsys profile /opt/flbench/venv38/bin/python3 device.py --client-id 0 --pipeline blond --ml-model densenet --data-dist local --experiment-name 2023-05-31_17:38_flbench_experiment-plan-baseline_blond_densenet_None_local_1_rounds_1_clients_0_dropout_nodp_0_prec_16
sleep 30
echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
sudo -E PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python nsys profile /opt/flbench/venv38/bin/python3 device.py --client-id 0 --pipeline shakespeare --ml-model lstm --data-dist local --experiment-name 2023-05-31_17:38_flbench_experiment-plan-baseline_shakespeare_lstm_None_local_1_rounds_1_clients_0_dropout_nodp_0_prec_16
