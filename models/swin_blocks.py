import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.swin_transformer import SwinTransformerBlock


def window_size_to_input_resolution(window_size: int) -> int:
    """Return the minimum spatial resolution that evenly tiles with
    *window_size* so that no padding is needed by the Swin block."""
    return window_size


class PatchEmbedding(nn.Module):
    """Project a feature map (B, C, H, W) → tokens (B, N, D) with a 1×1
    convolution followed by LayerNorm.  Learnable 1-D positional embeddings
    of shape (1, N, D) are added to the token sequence."""

    def __init__(self, in_channels: int, embed_dim: int, input_resolution: tuple[int, int]):
        super().__init__()
        self.init_H, self.init_W = input_resolution
        self.H, self.W = input_resolution
        self.N = self.init_H * self.init_W
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.N, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, N, D)"""
        x = self.proj(x)                                   # (B, D, H, W)
        self.H, self.W = x.shape[2], x.shape[3]
        x = rearrange(x, "b d h w -> b (h w) d")          # (B, N, D)
        x = self.norm(x)

        if x.shape[1] != self.pos_embed.shape[1]:
            pos_embed = self.pos_embed.reshape(
                1, self.init_H, self.init_W, self.embed_dim
            ).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed,
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).flatten(2).transpose(1, 2)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        return x

    def reshape_back(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, D) → (B, D, H, W) using stored H, W."""
        return rearrange(tokens, "b (h w) d -> b d h w", h=self.H, w=self.W)


def _tokens_to_spatial(tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(B, N, D) → (B, H, W, D) for timm SwinTransformerBlock."""
    return rearrange(tokens, "b (h w) d -> b h w d", h=H, w=W)


def _spatial_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """(B, H, W, D) → (B, N, D)."""
    return rearrange(x, "b h w d -> b (h w) d")


class RSTM(nn.Module):
    """Residual Swin Transformer Module – 6 Swin Transformer Blocks with a
    residual connection around the full sequence."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        input_resolution: tuple[int, int] = (56, 56),
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_embed = PatchEmbedding(dim, dim, input_resolution)

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
            )
            for i in range(6)
        ])

        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        tokens = self.patch_embed(x)           # (B, N, D)
        residual = tokens
        H, W = self.patch_embed.H, self.patch_embed.W

        out = _tokens_to_spatial(tokens, H, W)  # (B, H, W, D)
        for blk in self.blocks:
            out = blk(out)                       # (B, H, W, D)
        out = _spatial_to_tokens(out)            # (B, N, D)

        out = self.proj(out)                   # (B, N, D)
        out = out + residual                   # residual connection
        out = self.patch_embed.reshape_back(out)
        return out


class BSTM(nn.Module):
    """Bottleneck Swin Transformer Module – 12 Swin Transformer Blocks,
    followed by a 3×3 Conv that halves the channel count and a bilinear
    2× upsample.

    Returns:
        upsampled:  (B, dim//2, 2H, 2W)  – main output
        ds1:        (B, dim//2, H, W)     – pre-upsample features for
                                            deep supervision
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        input_resolution: tuple[int, int] = (7, 7),
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_embed = PatchEmbedding(dim, dim, input_resolution)

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
            )
            for i in range(12)
        ])

        self.channel_reduce = nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, H, W) → (upsampled, ds1)"""
        tokens = self.patch_embed(x)
        H, W = self.patch_embed.H, self.patch_embed.W

        out = _tokens_to_spatial(tokens, H, W)
        for blk in self.blocks:
            out = blk(out)
        out = _spatial_to_tokens(out)

        out = self.patch_embed.reshape_back(out)     # (B, dim, H, W)
        ds1 = self.channel_reduce(out)               # (B, dim//2, H, W)
        upsampled = self.upsample(ds1)               # (B, dim//2, 2H, 2W)
        return upsampled, ds1


class SDM(nn.Module):
    """Swin Decoder Module – 4 Swin Transformer Blocks followed by two 3×3
    Conv + BatchNorm + ReLU layers.

    Returns:
        conv_out:  (B, out_dim, H, W) – main output
        ds2:       (B, dim, H, W)     – tokens reshaped before final conv,
                                        for deep supervision
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        input_resolution: tuple[int, int] = (14, 14),
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_embed = PatchEmbedding(dim, dim, input_resolution)

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
            )
            for i in range(4)
        ])

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, H, W) → (conv_out, ds2)"""
        tokens = self.patch_embed(x)
        H, W = self.patch_embed.H, self.patch_embed.W

        out = _tokens_to_spatial(tokens, H, W)
        for blk in self.blocks:
            out = blk(out)
        out = _spatial_to_tokens(out)

        out = self.patch_embed.reshape_back(out)  # (B, dim, H, W)
        ds2 = out                                 # deep supervision tap
        out = self.conv1(out)
        conv_out = self.conv2(out)                # (B, out_dim, H, W)
        return conv_out, ds2
