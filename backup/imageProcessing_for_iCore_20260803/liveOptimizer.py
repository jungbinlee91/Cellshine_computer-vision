"""
liveOptimizer.py
----------------
LIVE 패널 전용 자동 밝기/대비(Auto Brightness/Contrast) 알고리즘.

ImageJ ContrastAdjuster.java 의 Auto B&C 와 동일한 히스토그램 기반 컷오프
방식 (saturation guard 포함):

  1) bincount 로 전체 히스토그램 (uint8 → 256 bins, uint16 → 1<<valid_bits bins)
  2) saturation guard: count > total/10 인 bin 은 무시
       (전체 픽셀의 10% 이상을 차지하는 단일 값 = 포화/배경. 컷오프 계산 제외)
  3) threshold = total / 5000: 이보다 많은 픽셀이 있는 bin 중 가장 왼쪽/오른쪽
       값을 low, high 로 사용
  4) LUT[v] = clip((v - low) * 255 / (high - low), 0, 255)
  5) 출력 = LUT[input]  (입력과 같은 shape, uint8)

valid_bits 자동 검출:
  16-bit TIFF 가 실제로는 12-bit 데이터일 수 있으므로 (max ≤ 4095),
  실측 최대값으로부터 ceil(log2(max+1)) 로 유효 비트수를 추정해 히스토그램
  bin 수를 최적화한다.
"""

import numpy as np

# ImageJ Auto B&C 상수 (ContrastAdjuster.java)
IMAGEJ_AUTO_THRESHOLD = 5000   # threshold = pixelCount / 5000
IMAGEJ_SAT_LIMIT_DIV = 10      # bins with count > N/10 are ignored


def imagej_auto_lut(arr: np.ndarray, bins: int) -> np.ndarray:
    """입력 배열의 히스토그램에서 양 끝 컷오프 → 8-bit LUT (length=bins) 반환."""
    if arr.dtype in (np.uint8, np.uint16):
        flat = arr.ravel()
        hist = np.bincount(flat, minlength=bins)
        if hist.size > bins:
            hist[bins - 1] += hist[bins:].sum()
            hist = hist[:bins]
    else:
        hist, _ = np.histogram(arr, bins=bins, range=(0, bins))

    total = arr.size
    threshold = total // IMAGEJ_AUTO_THRESHOLD
    limit = total // IMAGEJ_SAT_LIMIT_DIV
    counts = np.where(hist > limit, 0, hist)
    above = np.flatnonzero(counts > threshold)
    if above.size == 0:
        low, high = 0, bins - 1
    else:
        low, high = int(above[0]), int(above[-1])

    vals = np.arange(bins, dtype=np.float32)
    if high > low:
        scaled = (vals - low) * (255.0 / (high - low))
    else:
        scaled = vals
    return np.clip(scaled, 0, 255).astype(np.uint8)


def detect_valid_bits(arr: np.ndarray) -> int:
    """데이터 실측 최대값으로부터 유효 비트수를 추정."""
    if arr.dtype == np.uint8:
        return 8
    if arr.dtype == np.uint16:
        mx = int(arr.max()) if arr.size else 0
        if mx <= 0:
            return 16
        return max(8, int(np.ceil(np.log2(mx + 1))))
    return 16


class LiveOptimizer:
    """ImageJ Auto B&C 를 적용해 입력과 같은 shape 의 8-bit grayscale 반환."""

    def __init__(self, step: int = 10):
        # step: 과거 버전 호환용 (현 알고리즘은 풀 히스토그램 사용)
        self.step = step

    def run(self, img: np.ndarray, valid_bits: int | None = None) -> np.ndarray | None:
        if img is None:
            return None

        if img.dtype == np.uint8:
            lut = imagej_auto_lut(img, 256)
            return lut[img]

        if img.dtype == np.uint16:
            if valid_bits is None:
                valid_bits = detect_valid_bits(img)
            bins = 1 << valid_bits
            if int(img.max()) >= bins:
                img = np.minimum(img, bins - 1)
            lut = imagej_auto_lut(img, bins)
            return lut[img]

        # Fallback: uint16 으로 강제 변환 후 처리
        u16 = img.astype(np.uint16, copy=False)
        bins = 1 << detect_valid_bits(u16)
        lut = imagej_auto_lut(u16, bins)
        return lut[u16]
