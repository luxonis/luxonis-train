from pathlib import Path

import cv2
from torch import Tensor


def render_visualizations(
    visualizations: dict[str, dict[str, Tensor]], save_dir: str | Path | None
) -> None:
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(exist_ok=True, parents=True)

    i = 0
    for node_name, vzs in visualizations.items():
        for viz_name, viz_batch in vzs.items():
            for i, viz in enumerate(viz_batch):
                viz_arr = viz.detach().cpu().numpy().transpose(1, 2, 0)
                viz_arr = cv2.cvtColor(viz_arr, cv2.COLOR_RGB2BGR)
                name = f"{node_name}/{viz_name}/{i}"
                if save_dir is not None:
                    name = name.replace("/", "_")
                    cv2.imwrite(str(save_dir / f"{name}_{i}.png"), viz_arr)
                    i += 1
                else:
                    cv2.imshow(name, viz_arr)

    if save_dir is None:
        if cv2.waitKey(0) == ord("q"):
            exit()
