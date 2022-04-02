import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm

from easyrl.utils.rl_logger import logger


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 hidden_act=nn.ReLU,
                 output_act=None,
                 hid_layer_norm=False,
                 hid_spectral_norm=False,
                 out_layer_norm=False,
                 out_spectral_norm=False):
        super().__init__()
        if not isinstance(hidden_sizes, list):
            raise TypeError('hidden_sizes should be a list')
        in_size = input_size
        self.fcs = nn.ModuleList()
        for i, hid_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, hid_size)
            if hid_spectral_norm:
                fc = spectral_norm(fc)
            in_size = hid_size
            self.fcs.append(fc)
            if hid_layer_norm:
                self.fcs.append(nn.LayerNorm(hid_size))
            self.fcs.append(hidden_act())

        last_fc = nn.Linear(in_size, output_size)
        if out_spectral_norm:
            last_fc = spectral_norm(last_fc)
        self.fcs.append(last_fc)
        if out_layer_norm:
            self.fcs.append(nn.LayerNorm(output_size))
        if output_act is not None:
            self.fcs.append(output_act())

    def forward(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            x = torch.cat(x, dim=-1)
        for i, layer in enumerate(self.fcs):
            x = layer(x)
        return x
