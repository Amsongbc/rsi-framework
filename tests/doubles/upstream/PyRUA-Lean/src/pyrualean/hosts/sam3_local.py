"""Deterministic segmentation with the LocalSam3 interface (TEST DOUBLE, no model)."""
import dataclasses

import numpy as np


@dataclasses.dataclass
class SegmentResult:
    found: bool
    score: float
    box: list | None
    reason: str
    mask: np.ndarray | None


class LocalSam3:
    def segment(self, image, *, text_prompt=None, point=None, min_score=.2):
        red = np.asarray(image)[..., 0].astype(float)
        mask = red > red.mean() if red.std() > 0 else np.zeros(red.shape, bool)
        score = round(float(mask.mean()), 4)
        if not mask.any() or score < min_score:
            return SegmentResult(False, score, None, 'below_threshold', None)
        rows, cols = np.nonzero(mask)
        box = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
        return SegmentResult(True, score, box, 'synthetic_threshold', mask)
