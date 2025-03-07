from torch import Tensor, nn
from torch.nn import functional as F

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils import Packet

from .blocks import PPSegHead, SegHead


class MobileSegHead(BaseHead[Tensor, Tensor]):
    in_channels: list[int]
    attach_index: str = "all"

    task = Tasks.SEGMENTATION
    parser: str = "SegmentationParser"

    def __init__(
        self,
        backbone_indices,
        arm_out_chs,
        cm_bin_sizes,
        cm_out_ch,
        arm_type,
        resize_mode,
        use_last_fuse,
        seg_head_inter_chs,
        **kwargs,
    ):
        """MobileSeg segmentation head."""
        super().__init__(**kwargs)
        self.backbone_indices = backbone_indices

        backbone_out_chs = [self.in_channels[i] for i in backbone_indices]

        self.ppseg_head = PPSegHead(
            backbone_out_chs,
            arm_out_chs,
            cm_bin_sizes,
            cm_out_ch,
            arm_type,
            resize_mode,
            use_last_fuse,
        )

        self.seg_heads = nn.ModuleList()
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, self.n_classes))

    def forward(self, feats_backbone: list[Tensor]) -> list[Tensor]:
        x_hw = self.original_in_shape[1:]

        feats_selected = [feats_backbone[i] for i in self.backbone_indices]
        feats_head = self.ppseg_head(feats_selected)  # [..., x8, x16, x32]

        if self.training:
            logit_list = []
            for x, seg_head in zip(feats_head, self.seg_heads):
                x = seg_head(x)
                logit_list.append(x)
            logit_list = [
                F.interpolate(x, x_hw, mode="bilinear", align_corners=False)
                for x in logit_list
            ]
        else:
            x = self.seg_heads[0](feats_head[0])
            x = F.interpolate(x, x_hw, mode="bilinear", align_corners=False)
            logit_list = [x]

        return logit_list

    def wrap(
        self,
        output: list[Tensor],
    ) -> Packet[Tensor]:
        return {
            "features": output,
            "predictions": output[0],
        }
