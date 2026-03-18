import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.filters import threshold_otsu


class EAM(nn.Module):
    """Edge Attention Module – predicts an edge probability map from encoder
    features.  Returns raw logits (sigmoid is applied in the loss)."""

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(self, f1: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            f1: encoder feature map (B, C, H, W)
            target_size: (H_out, W_out) to resize the prediction to

        Returns:
            Edge probability logits (B, out_channels, H_out, W_out)
        """
        out = self.convs(f1)
        out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


class CAM(nn.Module):
    """Channel Attention Module – recalibrates channel-wise feature responses
    using parallel max-pool and avg-pool branches with shared MLPs
    (Equation 5: F' = M_ch(F) ⊗ F)."""

    def __init__(self, channels: int):
        super().__init__()
        mid = channels // 4

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Shared MLP (used by both branches)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: feature map (B, C, H, W)

        Returns:
            Channel-recalibrated feature map (B, C, H, W)
        """
        m_max = self.shared_mlp(self.max_pool(f))   # (B, C, 1, 1)
        m_avg = self.shared_mlp(self.avg_pool(f))   # (B, C, 1, 1)
        m_ch = torch.sigmoid(m_max + m_avg)          # (B, C, 1, 1)
        return m_ch * f


class FAM(nn.Module):
    """Feedback Attention Module – fuses the current feature map with the
    previous epoch's predicted mask to iteratively refine segmentation."""

    def __init__(self, in_channels: int):
        super().__init__()

        self.mask_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

        self.refine_path1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.refine_path2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        prev_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feature_map: (B, C, H, W) current encoder/decoder features
            prev_mask:   (B, 1, H_orig, W_orig) previous epoch prediction

        Returns:
            Concatenated refined features (B, 2C, H, W)
        """
        _, _, H, W = feature_map.shape

        # Step 1 – generate current binary mask from features
        current_soft = self.mask_generator(feature_map)            # (B, 1, H, W)
        current_binary_mask = (current_soft >= 0.5).float()        # (B, 1, H, W)

        # Step 2 – resize previous mask to feature resolution
        prev_mask_resized = F.adaptive_max_pool2d(prev_mask, (H, W))  # (B, 1, H, W)

        # Step 3 – union of current and previous masks
        union_mask = torch.maximum(current_binary_mask, prev_mask_resized)  # (B, 1, H, W)

        # Step 4 – enhance features with union mask
        enhanced_features = union_mask * feature_map               # (B, C, H, W)

        # Step 5 – parallel refinement
        refined1 = self.refine_path1(enhanced_features)            # (B, C, H, W)
        refined2 = self.refine_path2(feature_map)                  # (B, C, H, W)

        # Step 6 – concatenate
        return torch.cat([refined1, refined2], dim=1)              # (B, 2C, H, W)

    @staticmethod
    def get_initial_mask(images: np.ndarray) -> torch.Tensor:
        """Generate Otsu-thresholded masks for a batch of images (epoch 0).

        Args:
            images: (B, H, W, C) or (B, H, W) numpy array with dtype uint8

        Returns:
            Binary masks as float tensor (B, 1, H, W)
        """
        if images.ndim == 4:
            # Convert to grayscale: simple luminance
            gray = np.dot(images[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        else:
            gray = images

        masks = []
        for i in range(gray.shape[0]):
            img = gray[i]
            try:
                thresh = threshold_otsu(img)
            except ValueError:
                thresh = 128
            masks.append((img > thresh).astype(np.float32))

        masks = np.stack(masks, axis=0)[:, np.newaxis, :, :]  # (B, 1, H, W)
        return torch.from_numpy(masks)
