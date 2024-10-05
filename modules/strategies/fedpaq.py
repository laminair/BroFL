from typing import Dict

import numpy as np
import torch
import random

# import pdb


class FedPAQ(object):

    def __init__(self, s=5):
        """
        :param s: Tuning parameter for the degree of quantization. Higher s, less bits
        """
        self.state_dict = {}
        self.s = s

    def quantize(self, state_dict: dict) -> Dict:
        self.state_dict = state_dict
        for key in self.state_dict:
            if ".weight" in key:
                self.process_weight(self.state_dict[key])
            elif ".bias" in self.state_dict:
                self.process_bias(self.state_dict[key])

        return self.state_dict

    def process_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if len(weight.shape) == 1:  # e.g. batch normalization
            q_tensor = self.low_precision_quantizer(v=weight.cpu().detach().numpy(), s=self.s)
            return torch.from_numpy(q_tensor)

        elif len(weight.shape) == 2:  # e.g. weights of a fully connected layer
            m, n = weight.shape
            tens = weight.view(m * n, 1)  # flatten 2 d tensor of shape mxn into a vector mnx1
            tens = self.low_precision_quantizer(v=tens.cpu().detach().numpy(), s=self.s)
            return torch.from_numpy(tens.reshape(m, n))

        elif len(weight.shape) == 3:  # just in case, iterate over the 0 th index and combine the results at the end back to 3D
            empty_torch = torch.empty(weight.shape)
            iteration_idx = weight.shape[0]
            for i in range(iteration_idx):
                matrix_tensor = weight[i, :, :]
                m, n = matrix_tensor.shape
                tens = matrix_tensor.view(m * n, 1)  # flatten 2 d tensor of shape mxn into a vector mnx1
                tens = self.low_precision_quantizer(v=tens.cpu().detach().numpy(), s=self.s)
                empty_torch[i, :, :] = torch.from_numpy(tens.reshape(m, n))
            return empty_torch

        elif len(weight.shape) == 4:  # eg. convolutional layer
            empty_torch = torch.empty(weight.shape)
            iteration_idx1 = weight.shape[0]
            iteration_idx2 = weight.shape[1]
            for i in range(iteration_idx1):
                for j in range(iteration_idx2):
                    matrix_tensor = weight[i, j, :, :]
                    m, n = matrix_tensor.shape
                    tens = matrix_tensor.view(m * n, 1)  # flatten 2 d tensor of shape mxn into a vector mnx1
                    tens = self.low_precision_quantizer(v=tens.cpu().detach().numpy(), s=self.s)
                    empty_torch[i, j, :, :] = torch.from_numpy(tens.reshape(m, n))
            return empty_torch
            # pdb.set_trace()

    def process_bias(self, bias: torch.Tensor) -> torch.Tensor:
        if len(bias.shape) == 1:  # e.g. bias of a fully connected layer
            tens = self.low_precision_quantizer(v=bias.cpu().detach().numpy(), s=self.s)
            return torch.from_numpy(tens)

        elif len(bias.shape) == 2:  # just in case
            m, n = bias.shape
            tens = bias.view(m * n, 1)  # flatten 2 d tensor of shape mxn into a vector mnx1
            tens = self.low_precision_quantizer(v=tens.cpu().detach().numpy(), s=self.s)
            return torch.from_numpy(tens.reshape(m, n))

    @staticmethod
    def low_precision_quantizer(v: torch.Tensor, s: int = 5) -> np.array:
        """
        This functions implements the low precision quantizer Q_s(v) from
        https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf

        Args.
        v [numpy array]: vector to be quantized
        s [int] > 1: a tuning parameter, corresponding to the number of quantization levels => s uniformly levels
                     between 0 and 1.
        l [int]: 0 <= l < s such that |v_i|/||v||_2 element of [l/s, (l+1)/s]

        Return.
        Q_s(v) [numpy array]: quantized version of v.
        """
        if np.all(v == 0):  # if v is a zero vector, return a zero vector
            return v
        else:
            if len(v.shape) != 1:
                m = v.shape[0]  # this is a matrix of size mxn and this function works with array => reshape to 1xmn
                n = v.shape[1]
                v_reshaped = np.squeeze(np.reshape(v, (1, m * n)))
            else:  # no reshape
                n = 999  # placeholder
                v_reshaped = v
            v_quant = []
            l2 = np.linalg.norm(v_reshaped, 2)  # l2 norm
            a = []
            for i, elem in enumerate(v_reshaped):
                a.append(abs(elem) / l2)
                l = int((a[-1] * s))
                probability = random.random()
                ksi_probability = 1 - (a[-1] * s - l)
                if probability < ksi_probability:
                    ksi = l / s
                else:
                    ksi = (l + 1) / s
                elem_quant = l2 * np.sign(elem) * ksi
                v_quant.append(elem_quant)
            if n == 999:  # if we have not already reshaped
                return np.asarray(v_quant)
            else:
                return np.asarray(np.reshape(v_quant, (m, n)))

