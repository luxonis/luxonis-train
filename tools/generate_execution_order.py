from pathlib import Path

import torch
from cyclopts import App
from rich import print
from torch import nn

from luxonis_train import LuxonisModel

app = App()


def generate_execution_order(model: nn.Module) -> list[str]:
    order = []

    for name, module in model.named_modules():
        if list(module.parameters()):
            module.register_forward_hook(
                lambda mod, inp, out, n=name: order.append(n)
            )

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        model({"image": dummy_input})

    return order


@app.default
def main(
    opts: list[str] | None = None,
    /,
    *,
    config: str,
    weights: Path | None = None,
    output: Path | None = None,
) -> None:
    model = LuxonisModel(config, opts).lightning_module
    model.load_checkpoint(weights)
    order = generate_execution_order(model)
    if output:
        with open(output, "w") as f:
            f.writelines(f"{name}\n" for name in order)
    else:
        print(order)


if __name__ == "__main__":
    app()
