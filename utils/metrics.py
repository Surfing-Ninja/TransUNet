import numpy as np
import torch
from sklearn.metrics import accuracy_score
from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure
from skimage.metrics import structural_similarity


class SegmentationMetrics:
    """Compute a full suite of binary segmentation metrics for a single
    prediction / ground-truth pair."""

    @staticmethod
    def compute(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
        """
        Args:
            pred_mask: (H, W) binary {0, 1} predicted mask
            gt_mask:   (H, W) binary {0, 1} ground-truth mask

        Returns:
            dict with keys: dice, iou, sensitivity, specificity, precision,
            accuracy, mae, mean_hausdorff_distance, ssm,
            enhanced_alignment_measure
        """
        eps = 1e-6
        pred = pred_mask.astype(np.float64)
        gt = gt_mask.astype(np.float64)

        # ---- Basic counts ------------------------------------------------
        tp = np.sum(pred * gt)
        fp = np.sum(pred * (1 - gt))
        fn = np.sum((1 - pred) * gt)
        tn = np.sum((1 - pred) * (1 - gt))

        # ---- Core metrics ------------------------------------------------
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        sensitivity = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        precision = tp / (tp + fp + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        mae = np.mean(np.abs(pred - gt))

        # ---- Mean Hausdorff distance -------------------------------------
        mean_hd = SegmentationMetrics._mean_hausdorff(pred_mask, gt_mask)

        # ---- S-measure (SSM) ---------------------------------------------
        ssm = SegmentationMetrics._s_measure(pred, gt)

        # ---- Enhanced alignment measure ----------------------------------
        eam = SegmentationMetrics._enhanced_alignment(pred, gt)

        return {
            "dice":                     float(dice),
            "iou":                      float(iou),
            "sensitivity":              float(sensitivity),
            "specificity":              float(specificity),
            "precision":                float(precision),
            "accuracy":                 float(accuracy),
            "mae":                      float(mae),
            "mean_hausdorff_distance":  float(mean_hd),
            "ssm":                      float(ssm),
            "enhanced_alignment_measure": float(eam),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_hausdorff(
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> float:
        """Mean Hausdorff distance using morphological-gradient boundaries
        and EDT-based nearest-neighbour distances."""
        struct = generate_binary_structure(2, 1)

        pred_boundary = binary_dilation(pred, struct) ^ pred
        gt_boundary = binary_dilation(gt, struct) ^ gt

        # Handle degenerate cases
        if not np.any(pred_boundary) and not np.any(gt_boundary):
            return 0.0
        if not np.any(pred_boundary) or not np.any(gt_boundary):
            return float(np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2))

        # EDT from the complement gives distance to nearest boundary pixel
        dt_pred = distance_transform_edt(~pred_boundary)
        dt_gt = distance_transform_edt(~gt_boundary)

        # Mean distance: boundary(gt) → nearest boundary(pred) and reverse
        d_gt_to_pred = dt_pred[gt_boundary].mean()
        d_pred_to_gt = dt_gt[pred_boundary].mean()

        return float((d_gt_to_pred + d_pred_to_gt) / 2.0)

    @staticmethod
    def _s_measure(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
        """S-measure: α · object-aware similarity + (1−α) · region-aware
        similarity."""
        # Object-aware similarity: weighted mean of foreground & background IoU
        obj_sim = SegmentationMetrics._object_similarity(pred, gt)

        # Region-aware similarity: SSIM between the float arrays
        data_range = 1.0
        region_sim = structural_similarity(
            gt, pred, data_range=data_range,
        )

        return alpha * obj_sim + (1 - alpha) * region_sim

    @staticmethod
    def _object_similarity(pred: np.ndarray, gt: np.ndarray) -> float:
        """Weighted combination of foreground and background IoU."""
        eps = 1e-6
        fg_gt = gt.sum()
        bg_gt = (1 - gt).sum()
        total = fg_gt + bg_gt + eps

        # Foreground IoU
        inter_fg = (pred * gt).sum()
        union_fg = pred.sum() + fg_gt - inter_fg
        iou_fg = (inter_fg + eps) / (union_fg + eps)

        # Background IoU
        pred_bg = 1 - pred
        gt_bg = 1 - gt
        inter_bg = (pred_bg * gt_bg).sum()
        union_bg = pred_bg.sum() + bg_gt - inter_bg
        iou_bg = (inter_bg + eps) / (union_bg + eps)

        return float((fg_gt * iou_fg + bg_gt * iou_bg) / total)

    @staticmethod
    def _enhanced_alignment(pred: np.ndarray, gt: np.ndarray) -> float:
        """Enhanced alignment measure (Eq. 10 from the paper).

        E_φ = mean(1 − |pred_norm − gt_norm|)  where each map is
        normalised by subtracting its own mean.
        """
        eps = 1e-6

        pred_norm = pred - pred.mean()
        gt_norm = gt - gt.mean()

        # Alignment matrix
        align = 1.0 - np.abs(pred_norm - gt_norm)

        return float(align.mean())


class MetricAggregator:
    """Accumulate per-sample metric dicts and compute mean ± std."""

    def __init__(self):
        self._records: list[dict[str, float]] = []

    def reset(self) -> None:
        self._records.clear()

    def update(self, metrics: dict[str, float]) -> None:
        self._records.append(metrics)

    @property
    def count(self) -> int:
        return len(self._records)

    def mean_std(self) -> dict[str, tuple[float, float]]:
        """Return {metric_name: (mean, std)} over all accumulated samples."""
        if not self._records:
            return {}

        keys = self._records[0].keys()
        result = {}
        for k in keys:
            numeric_vals: list[float] = []
            for r in self._records:
                if k not in r:
                    continue
                try:
                    value = float(r[k])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    numeric_vals.append(value)

            if not numeric_vals:
                continue

            vals = np.array(numeric_vals, dtype=np.float64)
            result[k] = (float(vals.mean()), float(vals.std()))
        return result

    def summary(self) -> str:
        """Return a formatted multi-line summary string."""
        stats = self.mean_std()
        if not stats:
            return "No metrics recorded."

        lines = [
            f"Segmentation Metrics ({self.count} samples)",
            "-" * 50,
        ]
        for name, (mean, std) in stats.items():
            lines.append(f"  {name:30s}  {mean:.4f} ± {std:.4f}")
        lines.append("-" * 50)
        return "\n".join(lines)
