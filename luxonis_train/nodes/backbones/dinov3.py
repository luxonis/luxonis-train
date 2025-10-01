import torch

from typing import Literal, TypedDict


class DinoV3(BaseNode):
    def __init__(
            self,
            variant: Literal["vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16",
            "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
            ] = "vits16",
            **kwargs):
        super().__init__(**kwargs)
        self.backbone = self._get_backbone(variant)

    def forward(self, inputs: Tensor) -> list[Tensor]:
        return

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, DINOv3Kwargs]]:
        return "vitb16", {
            "vits16": {"variant": "vits16"},
            "vits16plus": {"variant": "vits16plus"},
            "vitb16": {"variant": "vitb16"},
            "vitl16": {"variant": "vitl16"},
            "vith16plus": {"variant": "vith16plus"},
            "vit7b16": {"variant": "vit7b16"},
            "convnext_tiny": {"variant": "convnext_tiny"},
            "convnext_small": {"variant": "convnext_small"},
            "convnext_base": {"variant": "convnext_base"},
            "convnext_large": {"variant": "convnext_large"},
        }

    @staticmethod
    def _get_backbone(
        variant: Literal[
            "vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16",
            "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
        ],
        weights: str,
        repo_dir: str = "facebookresearch/dinov3",
    ):
        variant_to_hub_name = {
            "vits16": "dinov3_vits16",
            "vits16plus": "dinov3_vits16plus",
            "vitb16": "dinov3_vitb16",
            "vitl16": "dinov3_vitl16",
            "vith16plus": "dinov3_vith16plus",
            "vit7b16": "dinov3_vit7b16",
            "convnext_tiny": "dinov3_convnext_tiny",
            "convnext_small": "dinov3_convnext_small",
            "convnext_base": "dinov3_convnext_base",
            "convnext_large": "dinov3_convnext_large",
        }

        if variant not in variant_to_hub_name:
            raise ValueError(f"Unsupported variant: {variant}")

        model = torch.hub.load(
            repo_or_dir=repo_dir,
            model=variant_to_hub_name[variant],
            source="github",
            weights=weights
        )
        return model