import torch.nn as nn


class AddNameWrapper(nn.Module):
    def __init__(self,
                 model,
                 name):
        super().__init__()
        self.model_name = name
        setattr(self, name, model)

    def forward(self, *args, **kwargs):
        return getattr(self, self.model_name)(*args, **kwargs)
