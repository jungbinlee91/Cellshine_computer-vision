from pathlib import Path

import cv2
import numpy as np
import tifffile
from tqdm import tqdm


# ==============================
# 사용자 설정
# ==============================

INPUT_DIR = Path("/mnt/d/홍보자료/20260720_R&D장비용/예시이미지") 
OUTPUT_VIDEO = INPUT_DIR / "result.mp4"

FPS = 15

LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0

# 0.5이면 가로·세로 절반, 전체 픽셀 수는 1/4
OUTPUT_SCALE = 0.2

# Percentile 계산용 간격
# 4이면 가로·세로 4픽셀마다 하나씩만 사용
PERCENTILE_SAMPLE_STEP = 8


def calculate_percentile_range(
    image: np.ndarray,
    low_percentile: float,
    high_percentile: float,
    sample_step: int,
) -> tuple[float, float]:
    """축소 샘플을 이용해 intensity 범위를 빠르게 계산합니다."""

    sampled = image[::sample_step, ::sample_step]

    low = float(np.percentile(sampled, low_percentile))
    high = float(np.percentile(sampled, high_percentile))

    if high <= low:
        high = low + 1.0

    return low, high


def stretch_to_uint8(
    image: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """지정한 intensity 범위를 0~255로 선형 변환합니다."""

    clipped = np.clip(image, low, high)

    # float 배열을 여러 개 만들지 않도록 OpenCV로 변환
    scale = 255.0 / (high - low)
    stretched = cv2.convertScaleAbs(
        clipped,
        alpha=scale,
        beta=-low * scale,
    )

    return stretched


def resize_frame(image: np.ndarray, scale: float) -> np.ndarray:
    """동영상 저장용 해상도로 축소합니다."""

    if scale == 1.0:
        return image

    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)

    # MP4 인코더 호환성을 위해 짝수 크기로 맞춤
    new_width -= new_width % 2
    new_height -= new_height % 2

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def main() -> None:
    tiff_files = sorted(
        list(INPUT_DIR.glob("*.tif"))
        + list(INPUT_DIR.glob("*.tiff"))
        + list(INPUT_DIR.glob("*.TIF"))
        + list(INPUT_DIR.glob("*.TIFF"))
    )

    if not tiff_files:
        raise RuntimeError(f"TIFF 파일이 없습니다: {INPUT_DIR}")

    first_image = tifffile.imread(tiff_files[0])

    if first_image.ndim != 2:
        raise ValueError(
            f"2차원 흑백 TIFF만 지원합니다. 현재 shape: {first_image.shape}"
        )

    first_frame = resize_frame(first_image, OUTPUT_SCALE)
    output_height, output_width = first_frame.shape

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        FPS,
        (output_width, output_height),
        isColor=False,
    )

    if not writer.isOpened():
        raise RuntimeError("VideoWriter를 열 수 없습니다.")

    try:
        for file_path in tqdm(
            tiff_files,
            desc="TIFF → MP4",
            unit="frame",
        ):
            image16 = tifffile.imread(file_path)

            low, high = calculate_percentile_range(
                image16,
                LOW_PERCENTILE,
                HIGH_PERCENTILE,
                PERCENTILE_SAMPLE_STEP,
            )

            image8 = stretch_to_uint8(image16, low, high)
            frame = resize_frame(image8, OUTPUT_SCALE)

            writer.write(frame)

    finally:
        writer.release()

    print(f"\n완료: {OUTPUT_VIDEO}")
    print(f"프레임 수: {len(tiff_files)}")
    print(f"영상 크기: {output_width} × {output_height}")
    print(f"FPS: {FPS}")


if __name__ == "__main__":
    main()