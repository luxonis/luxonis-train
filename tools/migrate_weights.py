from pathlib import Path
from typing import Any

import torch
from cyclopts import App
from rich import print

app = App()


def cleanup_name(name: str) -> str:
    try:
        keyword, _node_name, *rest = name.split(".")
    except ValueError:
        print(f"[red]Error processing name: {name}[/red]")
    assert keyword == "nodes"
    return ".".join(rest)


def migrate(
    old_weights: Path,
    old_execution_order: list[str],
    new_execution_order: list[str],
) -> dict[str, Any]:
    checkpoint = torch.load(old_weights, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    if len(old_execution_order) != len(new_execution_order):
        raise ValueError(
            "The old and new execution orders must have the same length."
        )
    old2new = dict(
        zip(
            map(cleanup_name, filter(bool, old_execution_order)),
            map(cleanup_name, filter(bool, new_execution_order)),
            strict=True,
        )
    )
    new_state_dict = {}
    for old_name, value in state_dict.items():
        *state_name_parts, parameter_name = old_name.split(".")
        state_name = ".".join(state_name_parts)
        if state_name in old2new:
            new_state = old2new[state_name]
        else:
            raise KeyError(
                f"No matching entry found in old2new for '{state_name}'"
            )
        new_name = f"{new_state}.{parameter_name}"
        new_state_dict[new_name] = value

    checkpoint["state_dict"] = new_state_dict
    return checkpoint


@app.default
def main(
    *,
    weights: Path,
    old_execution_order: Path,
    new_execution_order: Path,
    out_dir: Path = Path("migrated_weights"),
) -> None:
    old_order = old_execution_order.read_text().splitlines()
    new_order = new_execution_order.read_text().splitlines()
    new_weights = migrate(weights, old_order, new_order)
    torch.save(new_weights, out_dir / weights.name)


if __name__ == "__main__":
    app()
