import torch
from loguru import logger
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
            repo_dir=repo_dir,
            **kwargs
        )

        logger.warning(
            "DinoV3 is not convertible for RVC2. If RVC2 is your target platform, please pick a different backbone."
        )

    def forward(self, inputs: Tensor) -> list[Tensor]:
        x = self.backbone(inputs)
        return x

    @staticmethod
    def _get_weights_url(variant: str) -> str: # these links are unfortunately incorrect, the correct weights are links sent by META after filling a form
        """
        Returns the pretrained weights URL for a given DINOv3 variant.

        @param variant: The name of the DINOv3 variant.
        @return: URL string pointing to the pretrained weights.
        @raises ValueError: If the variant is not recognized.
        """
        base_url = "https://dl.fbaipublicfiles.com/dinov3"
        variant_urls = {
            "vits16": f"{base_url}/dinov3_vits16_pretrain.pth",
            "vits16plus": f"{base_url}/dinov3_vits16plus_pretrain.pth",
            "vitb16": f"{base_url}/dinov3_vitb16_pretrain.pth",
            "vitl16": f"{base_url}/dinov3_vitl16_pretrain.pth",
            "vith16plus": f"{base_url}/dinov3_vith16plus_pretrain.pth",
            "vit7b16": f"{base_url}/dinov3_vit7b16_pretrain.pth",
            "convnext_tiny": f"{base_url}/dinov3_convnext_tiny_pretrain.pth",
            "convnext_small": f"{base_url}/dinov3_convnext_small_pretrain.pth",
            "convnext_base": f"{base_url}/dinov3_convnext_base_pretrain.pth",
            "convnext_large": f"{base_url}/dinov3_convnext_large_pretrain.pth",
        }

        if variant not in variant_urls:
            raise ValueError(f"Unsupported variant '{variant}' for DINOv3 pretrained weights.")

        return variant_urls[variant]

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
            **kwargs
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
            weights=weights,
            **kwargs
        )
        return model
