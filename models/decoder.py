import torch
import torch.nn as nn
import torch.nn.functional as F

from models.swin_blocks import BSTM, SDM
from models.attention_modules import FAM


def _group_norm(num_channels: int) -> nn.GroupNorm:
    groups = min(32, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class DecoderBlock(nn.Module):
    """Two successive 3×3 Conv + BN + ReLU blocks that first concatenate
    the upsampled features with the skip connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            _group_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MaSDecoder(nn.Module):
    """MaS-TransUNet decoder: BSTM bottleneck → three DecoderBlocks with
    skip connections → FAM → SDM → segmentation head.

    Encoder skip shapes (for 224×224 input):
        skip1: (B, 256,  56, 56)
        skip2: (B, 512,  28, 28)
        skip3: (B, 1024, 14, 14)
        skip4: (B, 2048,  7,  7)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ---- BSTM bottleneck ---------------------------------------------
        # skip4 (2048, 7, 7) → upsampled (1024, 14, 14) + ds1 (1024, 7, 7)
        self.bstm = BSTM(
            dim=2048,
            embed_dim=config.bstm_embed_dim,
            num_heads=config.num_heads,
            window_size=config.window_size,
            input_resolution=(7, 7),
            depth=config.swin_bstm_depth,
            drop_rate=config.dropout,
        )

        # ---- Decoder blocks (concat channels → output channels) ----------
        # Stage 1: BSTM_out(1024) + skip3(1024) = 2048 → 512
        self.decoder1 = DecoderBlock(1024 + 1024, 512)
        # Stage 2: up(512) + skip2(512) = 1024 → 256
        self.decoder2 = DecoderBlock(512 + 512, 256)
        # Stage 3: up(256) + skip1(256) = 512 → 128
        self.decoder3 = DecoderBlock(256 + 256, 128)

        # ---- Bilinear 2× upsample used between decoder stages ------------
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # ---- FAM in the decoder ------------------------------------------
        self.fam = FAM(in_channels=128)
        # FAM doubles channels: 128 → 256; project back to 128 for SDM
        self.fam_proj = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            _group_norm(128),
            nn.ReLU(inplace=True),
        )

        # ---- SDM (Swin Decoder Module) -----------------------------------
        self.sdm = SDM(
            dim=128,
            out_dim=128,
            num_heads=config.num_heads,
            window_size=config.window_size,
            input_resolution=(56, 56),
            drop_rate=config.dropout,
        )

        # ---- Segmentation head -------------------------------------------
        # Concatenate decoder feature (128ch) with resized prev_mask (1ch).
        self.seg_head = nn.Conv2d(129, 1, kernel_size=1)

        # ---- Deep-supervision heads --------------------------------------
        self.ds1_head = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.ds2_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        encoder_outputs: dict[str, torch.Tensor],
        prev_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            encoder_outputs: dict from MaSEncoder with skip1–skip4 and edge_map
            prev_mask:       (B, 1, H_orig, W_orig) previous-epoch prediction

        Returns:
            dict with keys:
                pred_mask: (B, 1, H_in, W_in)  raw logits
                edge_map:  (B, 1, H_in, W_in)  raw logits (from encoder)
                ds1:       deep supervision output 1 from BSTM
                ds2:       deep supervision output 2 from SDM
        """
        skip1 = encoder_outputs["skip1"]      # (B,  256, 56, 56)
        skip2 = encoder_outputs["skip2"]      # (B,  512, 28, 28)
        skip3 = encoder_outputs["skip3"]      # (B, 1024, 14, 14)
        skip4 = encoder_outputs["skip4"]      # (B, 2048,  7,  7)

        # ---- BSTM bottleneck ---------------------------------------------
        bottleneck, ds1 = self.bstm(skip4)    # (B, 1024, 14, 14), (B, 1024, 7, 7)

        # ---- Decoder stage 1: concat with skip3 at 14×14 -----------------
        x = self.decoder1(bottleneck, skip3)  # (B, 512, 14, 14)

        # ---- Decoder stage 2: upsample → concat with skip2 at 28×28 -----
        x = self.up(x)                        # (B, 512, 28, 28)
        x = self.decoder2(x, skip2)           # (B, 256, 28, 28)

        # ---- Decoder stage 3: upsample → concat with skip1 at 56×56 -----
        x = self.up(x)                        # (B, 256, 56, 56)
        x = self.decoder3(x, skip1)           # (B, 128, 56, 56)

        # ---- Decoder FAM -------------------------------------------------
        x = self.fam(x, prev_mask)            # (B, 256, 56, 56)
        x = self.fam_proj(x)                  # (B, 128, 56, 56)

        # ---- SDM ---------------------------------------------------------
        x, ds2 = self.sdm(x)                  # (B, 128, 56, 56), (B, 128, 56, 56)

        # ---- Segmentation head -------------------------------------------
        prev_mask_resized = F.interpolate(
            prev_mask,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        x_cat = torch.cat([x, prev_mask_resized], dim=1)  # (B, 129, 56, 56)
        pred_mask = self.seg_head(x_cat)   # (B, 1, 56, 56) logits

        # ---- Deep supervision heads --------------------------------------
        ds1_out = self.ds1_head(ds1)   # (B, 1, 7, 7) logits
        ds2_out = self.ds2_head(ds2)   # (B, 1, 56, 56) logits

        # Upsample to input resolution
        target_size = encoder_outputs["edge_map"].shape[2:]
        pred_mask = F.interpolate(
            pred_mask, size=target_size, mode="bilinear", align_corners=False
        )

        return {
            "pred_mask": pred_mask,
            "edge_map": encoder_outputs["edge_map"],
            "ds1":      ds1_out,
            "ds2":      ds2_out,
        }
