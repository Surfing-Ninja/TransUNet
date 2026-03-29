import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class MaSLoss(nn.Module):
    """Combined loss for MaS-TransUNet.

    Total loss (Eq. 8):  L = Lp + Lb + L_ds1 + L_ds2

    where
        Lp   = primary loss on the final prediction      (Eq. 6)
        Lb   = boundary loss on the edge map              (Eq. 7)
        L_ds1, L_ds2 = deep-supervision primary losses
    """

    def __init__(self, config):
        super().__init__()
        self.lambda_weight = config.lambda_weight  # BCE weight in primary loss

    # ------------------------------------------------------------------
    # Private losses
    # ------------------------------------------------------------------

    def _weighted_iou_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted IoU loss.

        Args:
            pred:   (B, 1, H, W) probabilities in [0, 1]
            target: (B, 1, H, W) binary {0, 1}
        """
        smooth = 1e-6

        # Per-sample intersection & union over spatial dims (H, W)
        intersection = (pred * target).sum(dim=(2, 3))          # (B, 1)
        union = (pred + target - pred * target).sum(dim=(2, 3)) # (B, 1)

        iou = (intersection + smooth) / (union + smooth)        # (B, 1)
        return (1.0 - iou).mean()

    def _weighted_bce_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Class-weighted BCEWithLogits loss.

        Args:
            pred:   (B, 1, H, W) logits
            target: (B, 1, H, W) binary {0, 1}
        """
        target = target.float()
        eps = 1e-6
        pos_count = target.sum()
        neg_count = target.numel() - pos_count
        pos_weight = (neg_count / (pos_count + eps)).clamp(min=1.0, max=20.0).reshape(1)
        return F.binary_cross_entropy_with_logits(
            pred.float(),
            target,
            pos_weight=pos_weight,
            reduction="mean",
        )

    def _primary_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Primary segmentation loss (Eq. 6):  L_IoU + λ · L_BCE."""
        logits = pred.float()
        probs = torch.sigmoid(logits)
        iou = self._weighted_iou_loss(probs, target.float())
        bce = self._weighted_bce_loss(logits, target.float())
        return iou + self.lambda_weight * bce

    def _boundary_loss(
        self,
        pred_edge: torch.Tensor,
        gt_edge: torch.Tensor,
    ) -> torch.Tensor:
        """Distance-transform boundary loss (Kervadec-style supervision).

        Args:
            pred_edge: (B, 1, H, W) edge probability map
            gt_edge:   (B, 1, H, W) ground-truth edge probability map

        Uses an EDT map of the GT edges to penalize false positives farther
        away from the true boundary, plus a recall term on true edge pixels.
        This keeps gradients informative even for sparse edge maps.
        """
        gt_edge = (gt_edge >= 0.5).float()

        with torch.no_grad():
            gt_np = gt_edge.detach().cpu().numpy()
            dist_maps = []
            for i in range(gt_np.shape[0]):
                edge_i = gt_np[i, 0] > 0.5
                if edge_i.any():
                    dist_i = distance_transform_edt(~edge_i).astype(np.float32)
                    max_dist = float(dist_i.max())
                    if max_dist > 0.0:
                        dist_i = dist_i / max_dist
                else:
                    dist_i = np.zeros_like(gt_np[i, 0], dtype=np.float32)
                dist_maps.append(dist_i)

            dist_map = torch.from_numpy(np.stack(dist_maps, axis=0)).to(
                device=pred_edge.device,
                dtype=pred_edge.dtype,
            ).unsqueeze(1)

        false_positive_term = (pred_edge * (1.0 - gt_edge) * dist_map).mean()
        false_negative_term = ((1.0 - pred_edge) * gt_edge).mean()
        return false_positive_term + false_negative_term

    # ------------------------------------------------------------------
    # Helpers for deep supervision
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_ds(
        ds: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Resize single-channel deep-supervision logits to target size."""
        return F.interpolate(
            ds.float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            outputs: model output dict with keys
                pred_mask (B, 1, H, W) raw logits
                edge_map  (B, 1, H, W) raw logits
                ds1       (B, 1, h1, w1) raw logits
                ds2       (B, 1, h2, w2) raw logits
            targets: dict with keys
                mask  (B, 1, H, W) binary ground-truth mask
                edge  (B, 1, H, W) ground-truth edge map

        Returns:
            total_loss:  scalar tensor
            loss_dict:   dict of individual loss components for logging
        """
        gt_mask = targets["mask"]
        gt_edge = targets["edge"]
        target_size = (gt_mask.shape[2], gt_mask.shape[3])

        # Primary loss on final prediction (Eq. 6)
        lp = self._primary_loss(outputs["pred_mask"], gt_mask)

        # Boundary loss on edge map (Eq. 7)
        pred_edge = torch.sigmoid(outputs["edge_map"])
        lb = self._boundary_loss(pred_edge, gt_edge)

        # Deep-supervision losses
        ds1_pred = self._prepare_ds(outputs["ds1"], target_size)
        lds1 = self._primary_loss(ds1_pred, gt_mask)

        ds2_pred = self._prepare_ds(outputs["ds2"], target_size)
        lds2 = self._primary_loss(ds2_pred, gt_mask)

        total_loss = lp + lb + lds1 + lds2

        loss_dict = {
            "Lp":   lp.detach(),
            "Lb":   lb.detach(),
            "Lds1": lds1.detach(),
            "Lds2": lds2.detach(),
        }

        return total_loss, loss_dict
