import torch


class TripleLRSGD:
    def __init__(self, model: torch.nn.Module, params: dict) -> None:
        """TripleLRSGD is a custom optimizer that separates weights into
        batch norm weights, regular weights, and biases.

        @type model: torch.nn.Module
        @param model: The model to be used
        @type params: dict
        @param params: The parameters to be used for the optimizer
        """
        self.model = model
        self.params = {
            "lr": 0.02,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "nesterov": True,
        }
        if params:
            self.params.update(params)

    def create_optimizer(self):
        batch_norm_weights, regular_weights, biases = [], [], []

        for module in self.model.modules():
            if hasattr(module, "bias") and isinstance(
                module.bias, torch.nn.Parameter
            ):
                biases.append(module.bias)
            if isinstance(module, torch.nn.BatchNorm2d):
                batch_norm_weights.append(module.weight)
            elif hasattr(module, "weight") and isinstance(
                module.weight, torch.nn.Parameter
            ):
                regular_weights.append(module.weight)

        optimizer = torch.optim.SGD(
            [
                {
                    "params": batch_norm_weights,
                    "lr": self.params["lr"],
                    "momentum": self.params["momentum"],
                    "nesterov": self.params["nesterov"],
                },
                {
                    "params": regular_weights,
                    "weight_decay": self.params["weight_decay"],
                },
                {"params": biases},
            ],
            lr=self.params["lr"],
            momentum=self.params["momentum"],
            nesterov=self.params["nesterov"],
        )

        return optimizer
