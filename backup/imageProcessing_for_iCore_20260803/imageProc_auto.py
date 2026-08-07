import cv2
import numpy as np


def local_contrast_normalization(
    image: np.ndarray,
    local_sigma: float = 15.0,
    noise_floor: float = 3.0,
) -> np.ndarray:
    """
    주변 영역의 평균과 표준편차를 이용한 local normalization.

    계산식:
        local_mean = GaussianBlur(image)
        local_variance = GaussianBlur(image²) - local_mean²
        normalized = (image - local_mean) / max(local_std, noise_floor)
    """
    # 이미 float32이면 불필요한 복사와 형변환을 하지 않음
    if image.dtype == np.float32:
        img = image
    else:
        img = image.astype(np.float32)

    # E[x]
    local_mean = cv2.GaussianBlur(
        img,
        (0, 0),
        sigmaX=local_sigma,
        sigmaY=local_sigma,
    )

    # x² 계산용 배열
    img_sq = np.empty_like(img)
    np.multiply(img, img, out=img_sq)

    # E[x²]
    local_mean_sq = cv2.GaussianBlur(
        img_sq,
        (0, 0),
        sigmaX=local_sigma,
        sigmaY=local_sigma,
    )

    # img_sq 배열을 local_mean² 저장 공간으로 재사용
    np.multiply(local_mean, local_mean, out=img_sq)

    # variance = E[x²] - E[x]²
    # local_mean_sq 배열을 variance 저장 공간으로 재사용
    np.subtract(
        local_mean_sq,
        img_sq,
        out=local_mean_sq,
    )

    np.maximum(
        local_mean_sq,
        0.0,
        out=local_mean_sq,
    )

    # variance 배열을 std 배열로 재사용
    np.sqrt(
        local_mean_sq,
        out=local_mean_sq,
    )

    # std = max(std, noise_floor)
    np.maximum(
        local_mean_sq,
        noise_floor,
        out=local_mean_sq,
    )

    # normalized = img - local_mean
    normalized = np.empty_like(img)

    np.subtract(
        img,
        local_mean,
        out=normalized,
    )

    # normalized /= max(local_std, noise_floor)
    np.divide(
        normalized,
        local_mean_sq,
        out=normalized,
    )

    return normalized


def normalized_to_uint8(
    normalized: np.ndarray,
    lower: float = -2.0,
    upper: float = 5.0,
) -> np.ndarray:
    """
    local z-score 결과의 고정 범위를 0~255로 변환.
    """
    if upper <= lower:
        raise ValueError("upper는 lower보다 커야 합니다.")

    # 새 float 배열 하나만 생성
    scaled = np.clip(normalized, lower, upper)

    # 이후 계산은 같은 배열을 계속 재사용
    np.subtract(
        scaled,
        lower,
        out=scaled,
    )

    np.multiply(
        scaled,
        255.0 / (upper - lower),
        out=scaled,
    )

    np.rint(
        scaled,
        out=scaled,
    )

    return scaled.astype(np.uint8)


class ImageProcessor_Auto:
    # CLAHE 객체 생성 비용을 줄이기 위한 캐시
    _clahe = None
    _clahe_clip_limit = None
    _clahe_tile_grid_size = (8, 8)

    @classmethod
    def _get_clahe(
        cls,
        clip_limit: float,
    ):
        """
        같은 설정이면 기존 CLAHE 객체를 재사용.
        """
        if (
            cls._clahe is None
            or cls._clahe_clip_limit != clip_limit
        ):
            cls._clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=cls._clahe_tile_grid_size,
            )
            cls._clahe_clip_limit = clip_limit

        return cls._clahe

    @classmethod
    def apply_adaptive_enhancement(
        cls,
        image: np.ndarray | None,
        denoise_sigma: float = 3.0,
        local_sigma: float = 100.0,
        noise_floor: float = 5.0,
        output_low: float = -2.0,
        output_high: float = 5.0,
        clahe_clip_limit: float = 1.5,
    ) -> np.ndarray | None:
        """
        영상별 밝기와 국소 배경 변화에 적응하는 전처리.

        1. Gaussian denoising
        2. Local mean/std normalization
        3. 고정 z-score 범위로 8-bit 변환
        4. CLAHE
        """
        if image is None:
            return None

        if image.ndim != 2:
            raise ValueError(
                "image는 2차원 grayscale 영상이어야 합니다."
            )

        # 입력이 이미 float32이면 불필요한 복사 방지
        if image.dtype == np.float32:
            img = image
        else:
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
            clahe = cls._get_clahe(clahe_clip_limit)
            output = clahe.apply(output)

        return output