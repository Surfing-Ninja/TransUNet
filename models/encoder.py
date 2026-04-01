import torch
import torch.nn as nn
import torchvision.models as models

from models.swin_blocks import RSTM
from models.attention_modules import CAM, FAM, EAM


def _group_norm(num_channels: int) -> nn.GroupNorm:
    groups = min(32, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class MaSEncoder(nn.Module):
    """MaS-TransUNet encoder: ResNet-50 backbone augmented with FAM, CAM,
    RSTM, and EAM modules at each stage."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ---- ResNet-50 backbone (pretrained) ----------------------------
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Initial block: conv1 → bn1 → relu → maxpool  (3 → 64, /4, 56×56)
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        # Residual stages
        self.layer1 = resnet.layer1   # 64  → 256,  56×56
        self.layer2 = resnet.layer2   # 256 → 512,  28×28
        self.layer3 = resnet.layer3   # 512 → 1024, 14×14
        self.layer4 = resnet.layer4   # 1024→ 2048,  7×7

        # ---- FAM after initial block ------------------------------------
        self.fam = FAM(in_channels=64)
        # FAM outputs 2×64 = 128 channels; project back to 64 for layer1
        self.fam_proj = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            _group_norm(64),
            nn.ReLU(inplace=True),
        )

        # ---- Stage 1: CAM + RSTM (256 ch, 56×56) -----------------------
        # embed_dim = dim (no bottleneck, matches paper exactly)
        self.cam1 = CAM(channels=256)
        self.rstm1 = RSTM(
            dim=256,
            num_heads=config.num_heads,
            window_size=config.window_size,
            input_resolution=(56, 56),
            drop_rate=config.dropout,
        )

        # ---- Stage 2: CAM + RSTM (512 ch, 28×28) -----------------------
        self.cam2 = CAM(channels=512)
        self.rstm2 = RSTM(
            dim=512,
            num_heads=config.num_heads,
            window_size=config.window_size,
            input_resolution=(28, 28),
            drop_rate=config.dropout,
        )

        # ---- Stage 3: CAM + RSTM (1024 ch, 14×14) ----------------------
        self.cam3 = CAM(channels=1024)
        self.rstm3 = RSTM(
            dim=1024,
            num_heads=config.num_heads,
            window_size=config.window_size,
            input_resolution=(14, 14),
            drop_rate=config.dropout,
        )

        # ---- EAM on Block-1 output (256 ch) → edge map -----------------
        self.eam = EAM(in_channels=256)

    def forward(
        self,
        image: torch.Tensor,
        prev_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            image:     (B, 3, 224, 224)
            prev_mask: (B, 1, H_orig, W_orig)  previous-epoch prediction

        Returns:
            dict with keys:
                skip1:    (B, 256,  56, 56)
                skip2:    (B, 512,  28, 28)
                skip3:    (B, 1024, 14, 14)
                skip4:    (B, 2048,  7,  7)
                edge_map: (B, 1, 224, 224)  raw logits
        """
        target_size = (image.shape[2], image.shape[3])

        # Initial block + FAM
        x = self.initial(image)                 # (B,  64, 56, 56)
        x = self.fam(x, prev_mask)              # (B, 128, 56, 56)
        x = self.fam_proj(x)                    # (B,  64, 56, 56)

        # Stage 1
        x = self.layer1(x)                      # (B, 256, 56, 56)
        x = self.cam1(x)
        skip1 = self.rstm1(x)                   # (B, 256, 56, 56)

        # Edge map from Block-1 features
        edge_map = self.eam(skip1, target_size)  # (B, 1, 224, 224)

        # Stage 2
        x = self.layer2(skip1)                   # (B, 512, 28, 28)
        x = self.cam2(x)
        skip2 = self.rstm2(x)                    # (B, 512, 28, 28)

        # Stage 3
        x = self.layer3(skip2)                   # (B, 1024, 14, 14)
        x = self.cam3(x)
        skip3 = self.rstm3(x)                    # (B, 1024, 14, 14)

        # Stage 4 (no CAM / RSTM)
        skip4 = self.layer4(skip3)               # (B, 2048, 7, 7)

        return {
            "skip1":    skip1,
            "skip2":    skip2,
            "skip3":    skip3,
            "skip4":    skip4,
            "edge_map": edge_map,
        }
