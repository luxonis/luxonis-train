from torch import Tensor, nn


class HSigmoid(nn.Module):
    def __init__(self):
        """Hard-Sigmoid (approximated sigmoid) activation function from
        U{Searching for MobileNetV3<https://arxiv.org/abs/1905.02244>}."""
        super().__init__()
        self.relu = nn.ReLU6(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + 3) / 6
