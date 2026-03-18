import torch
import torch.nn as nn

from models.encoder import MaSEncoder
from models.decoder import MaSDecoder


class MaSTransUNet(nn.Module):
    """MaS-TransUNet: Multi-scale Attention and Swin Transformer UNet
    for medical image segmentation."""

    def __init__(self, config):
        super().__init__()
        self.encoder = MaSEncoder(config)
        self.decoder = MaSDecoder(config)

    def forward(
        self,
        image: torch.Tensor,
        prev_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            image:     (B, 3, H, W)
            prev_mask: (B, 1, H, W)

        Returns:
            dict with keys: pred_mask, edge_map, ds1, ds2
        """
        encoder_outputs = self.encoder(image, prev_mask)
        output = self.decoder(encoder_outputs, prev_mask)
        return output


def get_model_info(config=None):
    """Instantiate the model, run a dummy forward pass, and print summary."""
    if config is None:
        from config import CFG
        config = CFG

    device = config.device
    model = MaSTransUNet(config).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params / 1e6:.2f}M")

    dummy_image = torch.randn(1, 3, 224, 224, device=device)
    dummy_mask = torch.zeros(1, 1, 224, 224, device=device)

    with torch.no_grad():
        out = model(dummy_image, dummy_mask)

    for key, val in out.items():
        print(f"  {key:10s}: {tuple(val.shape)}")


def build_model(config) -> MaSTransUNet:
    """Build MaSTransUNet and move to the configured device."""
    return MaSTransUNet(config).to(config.device)
