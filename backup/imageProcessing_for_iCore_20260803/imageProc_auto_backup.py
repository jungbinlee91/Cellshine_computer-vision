import cv2
import numpy as np


def local_contrast_normalization(
    image: np.ndarray,
    local_sigma: float = 15.0,
    noise_floor: float = 3.0,
) -> np.ndarray:
    """
    주변 영역의 평균과 표준편차를 이용한 local normalization.

    Parameters
    ----------
    image:
        2D grayscale image
    local_sigma:
        국소 통계를 계산하는 공간 범위.
        세포 지름보다 충분히 크게 설정하는 것이 좋음.
    noise_floor:
        표준편차가 매우 작은 영역에서 노이즈가 과증폭되는 것을 방지.
    """
    img = image.astype(np.float32)

    local_mean = cv2.GaussianBlur(
        img,
        (0, 0),
        sigmaX=local_sigma,
        sigmaY=local_sigma,
    )

    local_mean_sq = cv2.GaussianBlur(
        img * img,
        (0, 0),
        sigmaX=local_sigma,
        sigmaY=local_sigma,
    )

    local_variance = np.maximum(
        local_mean_sq - local_mean * local_mean,
        0.0,
    )
    local_std = np.sqrt(local_variance)

    normalized = (img - local_mean) / np.maximum(local_std, noise_floor)

    return normalized

def normalized_to_uint8(
    normalized: np.ndarray,
    lower: float = -2.0,
    upper: float = 5.0,
) -> np.ndarray:
    """
    local z-score 결과의 고정 범위를 0~255로 변환.
    """
    clipped = np.clip(normalized, lower, upper)
    scaled = (clipped - lower) / (upper - lower) * 255.0
    return np.round(scaled).astype(np.uint8)

class ImageProcessor_Auto:
    @staticmethod
    def apply_adaptive_enhancement(
        image: np.ndarray | None,
        denoise_sigma: float = 3.0, # default 1
        local_sigma: float = 100.0, # default 15
        noise_floor: float = 5.0, # default 3
        output_low: float = -2.0, # default -2
        output_high: float = 5.0, # default 5
        clahe_clip_limit: float = 1.5, # default 1.5
    ) -> np.ndarray | None:
        """
        영상별 밝기와 국소 배경 변화에 적응하는 전처리.

        1. 약한 Gaussian denoising
        2. Local mean/std normalization
        3. 고정 z-score 범위로 8-bit 변환
        4. 약한 CLAHE
        """
        if image is None:
            return None

        if image.ndim != 2:
            raise ValueError("image는 2차원 grayscale 영상이어야 합니다.")

        img = image.astype(np.float32)

        if denoise_sigma > 0:
            img = cv2.GaussianBlur(
                img,
                (0, 0),
                sigmaX=denoise_sigma,
                sigmaY=denoise_sigma,
            )

        normalized = local_contrast_normalization(
            img,
            local_sigma=local_sigma,
            noise_floor=noise_floor,
        )

        output = normalized_to_uint8(
            normalized,
            lower=output_low,
            upper=output_high,
        )

        if clahe_clip_limit > 0:
            clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=(8, 8),
            )
            output = clahe.apply(output)

        return output