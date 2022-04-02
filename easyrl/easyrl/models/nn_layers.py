import torch
import torch.nn as nn


class SoftArgmax2D(nn.Module):
    """
    Also called spatial softmax
    """

    def __init__(self, softmax_temp=None):
        super().__init__()
        self.softmax = nn.Softmax(dim=2)
        if softmax_temp is None:
            self.softmax_temp = nn.Parameter(torch.ones(1))
        else:
            self.softmax_temp = torch.ones(1) * softmax_temp

    def forward(self, x):
        """

        Args:
            x (torch.tensor): input image features,
                shape: [N, C, H, W]

        Returns:
            torch.tensor: expected pixel coordinate ranging
            from -1 to 1 of the soft-max point, shape: [N, C, 2]

        """
        N, C, H, W = x.shape
        x_flat = x.view((N, C, H * W)) / self.softmax_temp
        x_softmax = self.softmax(x_flat)
        prob = x_softmax.view(N, C, H, W)
        ti, tj = torch.meshgrid(torch.linspace(-1, 1, H),
                                torch.linspace(-1, 1, W))
        ti = ti.to(prob.device)
        tj = tj.to(prob.device)
        pos_i = prob * ti
        pos_j = prob * tj
        expected_i = pos_i.sum([-2, -1])
        expected_j = pos_j.sum([-2, -1])
        expected_xy = torch.stack([expected_i, expected_j], 2)
        return expected_xy
