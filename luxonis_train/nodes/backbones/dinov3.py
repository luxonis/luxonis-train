import torch

from typing import Literal, TypedDict


class DinoV3(BaseNode):
    def __init__(
        self,
        variant: Literal[
            "vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16",
            "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
        ] = "vitb16",
        weights: Literal["download", "none"] | None = "download",
        repo_dir: str = "facebookresearch/dinov3",
        **kwargs):
        """DinoV3 backbone

        Source: U{https://github.com/facebookresearch/dinov3}

        @param variant: Architecture variant of the DINOv3 backbone.
        @type variant: Literal of supported DINOv3 variants.

        @param weights: Whether to download pretrained weights ("download") or initialize randomly ("none").
        @type weights: Literal["download", "none"] or None

        @param repo_dir: Torch Hub repo to use. Defaults to the official "facebookresearch/dinov3".
        @type repo_dir: str
        """
        super().__init__(**kwargs)

        if weights == "download":
            weights_url = self._get_weights_url(variant)
        else:
            weights_url = None

        self.backbone = self._get_backbone(
            variant=variant,
            weights=weights_url,
            repo_dir=repo_dir
        )

    def forward(self, inputs: Tensor) -> list[Tensor]:
        x = self.backbone(inputs)
        return [x]

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
