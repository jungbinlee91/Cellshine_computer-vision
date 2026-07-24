from pathlib import Path
import os
import cv2
import imageio.v2 as imageio
import numpy as np
import tifffile
from tqdm import tqdm
from pygifsicle import optimize
from PIL import Image

# ==============================
# 사용자 설정
# ==============================

INPUT_DIR = Path("/mnt/d/IR자료/20260720_R&D장비용/예시이미지") 
OUTPUT_GIF = INPUT_DIR / "result_optimized.gif"

# 앞에서 제외할 프레임 수
SKIP_FIRST_FRAMES = 30
# 뒤에서 제외할 프레임 수
SKIP_LAST_FRAMES = 0

IMAGE_MODE = "grayscale" # "grayscale" , "color"
GRAYSCALE_COLORS = 32
COLOR_COLORS = 128

FPS = 15
FRAME_STEP = 2
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.8

# 0.5이면 가로·세로 절반, 전체 픽셀 수는 1/4
OUTPUT_SCALE = 0.075

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

def find_tiff_files(input_dir: Path) -> list[Path]:
    """대소문자를 구분하지 않고 TIFF 파일을 찾습니다."""

    return sorted(
        file_path
        for file_path in input_dir.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in {".tif", ".tiff"}
    )

def resize_frame(
    image: np.ndarray,
    scale: float,
) -> np.ndarray:
    """GIF 저장용 해상도로 이미지를 축소합니다."""

    if scale <= 0:
        raise ValueError("OUTPUT_SCALE은 0보다 커야 합니다.")

    if scale == 1.0:
        return image

    new_width = max(1, round(image.shape[1] * scale))
    new_height = max(1, round(image.shape[0] * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def main() -> None:
    if FPS <= 0:
        raise ValueError("FPS는 0보다 커야 합니다.")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {INPUT_DIR}")

    tiff_files = find_tiff_files(INPUT_DIR)

    total_frames = len(tiff_files)
    if SKIP_FIRST_FRAMES + SKIP_LAST_FRAMES >= total_frames:
        raise ValueError(
            "SKIP_FIRST_FRAMES + SKIP_LAST_FRAMES가 "
            "전체 프레임 수보다 크거나 같습니다."
        )
    tiff_files = tiff_files[
        SKIP_FIRST_FRAMES : total_frames - SKIP_LAST_FRAMES
    ]
    tiff_files = tiff_files[::FRAME_STEP]

    if not tiff_files:
        raise RuntimeError(f"TIFF 파일이 없습니다: {INPUT_DIR}")

    first_image = tifffile.imread(tiff_files[0])

    if first_image.ndim != 2:
        raise ValueError(
            "2차원 흑백 TIFF만 지원합니다. "
            f"현재 이미지 shape: {first_image.shape}"
        )

    first_frame = resize_frame(first_image, OUTPUT_SCALE)
    output_height, output_width = first_frame.shape

    OUTPUT_GIF.parent.mkdir(parents=True, exist_ok=True)

    # 각 프레임이 표시되는 시간(초)
    frame_duration = 1.0 / FPS

    with imageio.get_writer(
        OUTPUT_GIF,
        mode="I",
        duration=frame_duration,
        loop=0,  # 0이면 무한 반복
    ) as writer:

        for file_path in tqdm(
            tiff_files,
            desc="TIFF → GIF",
            unit="frame",
        ):
            image16 = tifffile.imread(file_path)

            if IMAGE_MODE == "grayscale":
                if image16.ndim != 2:
                    raise ValueError(
                        f"흑백 모드에서는 2차원 TIFF만 지원합니다: "
                        f"{file_path.name}, shape={image16.shape}"
                    )

            elif IMAGE_MODE == "color":
                if image16.ndim != 3 or image16.shape[2] not in (3, 4):
                    raise ValueError(
                        f"컬러 모드에서는 RGB 또는 RGBA TIFF만 지원합니다: "
                        f"{file_path.name}, shape={image16.shape}"
                    )

            low, high = calculate_percentile_range(
                image16,
                LOW_PERCENTILE,
                HIGH_PERCENTILE,
                PERCENTILE_SAMPLE_STEP,
            )

            image8 = stretch_to_uint8(
                image16,
                low,
                high,
            )
            
            frame = resize_frame(
                image8,
                OUTPUT_SCALE,
            )
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

            if frame.shape != (output_height, output_width):
                raise ValueError(
                    f"이미지 크기가 서로 다릅니다: {file_path.name}, "
                    f"현재={frame.shape}, 기준="
                    f"{(output_height, output_width)}"
                )

            # ==============================
            # GIF 색상 최적화
            # ==============================

            frame_pil = Image.fromarray(frame)

            if IMAGE_MODE == "grayscale":
                frame_pil = frame_pil.convert("L")

                frame_pil = frame_pil.quantize(
                    colors=GRAYSCALE_COLORS,
                    method=Image.MEDIANCUT,
                    dither=Image.Dither.NONE,
                )

            elif IMAGE_MODE == "color":
                frame_pil = frame_pil.convert("RGB")

                frame_pil = frame_pil.quantize(
                    colors=COLOR_COLORS,
                    method=Image.MEDIANCUT,
                    dither=Image.Dither.FLOYDSTEINBERG,
                )

            else:
                raise ValueError(
                    'IMAGE_MODE는 "grayscale" 또는 "color"여야 합니다.'
                )
            writer.append_data(frame)
    ## optimize ##
    before = os.path.getsize(OUTPUT_GIF)
    optimize(str(OUTPUT_GIF))
    after = os.path.getsize(OUTPUT_GIF)
    print(f"Before : {before/1024/1024:.2f} MB")
    print(f"After  : {after/1024/1024:.2f} MB")

    print(f"\n완료: {OUTPUT_GIF}")
    print(f"프레임 수: {len(tiff_files)}")
    print(f"GIF 크기: {output_width} × {output_height}")
    print(f"재생 속도: {FPS} FPS")
    print("반복 재생: 무한 반복")


if __name__ == "__main__":
    main()