"""
imageProc.py
------------
PROCESSING 패널 전용 이미지 강조 알고리즘.

파이프라인 (입력: 8/16-bit grayscale, 출력: 8-bit grayscale):

  1) Gaussian Blur (5×5 커널, σ = blur_sigma)
     - 고주파 노이즈 억제

  2) 퍼센타일 선형 스트레칭 (p_low% ~ p_high%)
     - 10×10 서브샘플 히스토그램에서 두 백분위 값을 잘라내
       해당 구간을 [0, 255] 8-bit 로 선형 매핑

  3) CLAHE (Contrast-Limited Adaptive Histogram Equalization)
     - 8×8 타일, clipLimit = clip_limit
     - 국소 콘트라스트 향상으로 술잔세포 가장자리 강조
"""

import cv2
import numpy as np


def _stretched_8bit(target: np.ndarray, low_p: float, high_p: float) -> np.ndarray:
    """10x10 서브샘플에서 [low_p, high_p] 백분위 구간을 0..255 로 선형 매핑."""
    sample = target[::10, ::10]
    low, high = np.percentile(sample, [low_p, high_p])
    if high > low:
        alpha = 255.0 / (high - low)
        beta = -low * alpha
        return cv2.convertScaleAbs(target, alpha=alpha, beta=beta)
    return cv2.normalize(target, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


class ImageProcessor:
    @staticmethod
    def apply_enhancement(
        image: np.ndarray,
        p_low: float = 1.0,
        p_high: float = 99.9,
        blur_sigma: float = 5.0,
        clip_limit: float = 25.0,
    ) -> np.ndarray | None:
        """Blur → percentile stretch → CLAHE → 8-bit grayscale."""
        if image is None:
            return None

        img_f = image.astype(np.float32)
        if blur_sigma > 0:
            img_f = cv2.GaussianBlur(img_f, (0, 0), sigmaX=blur_sigma)

        stretched = _stretched_8bit(img_f, p_low, p_high)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(stretched)
