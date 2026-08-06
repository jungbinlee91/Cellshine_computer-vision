"""
detectingGC.py
--------------
DETECTING 패널의 술잔세포(Goblet Cell) 검출 알고리즘.

파이프라인 (입력: PROCESSING 결과 8-bit grayscale + ROI 좌표):

  1) ROI 추출
  2) Median Blur (5×5) — 점잡음 제거
  3) Adaptive Threshold (Gaussian, block_size_px, adaptive_c)
     → 지역적 이진화
  4) Watershed 분할 (distance transform + connectedComponents)
     → 군집된 점들을 개별 객체로 분리
  5) 컨투어 후처리 필터:
       - 면적 [min_size_px, max_size_px] 범위 내
       - 원형도(circularity) ≥ min_circularity
       - 컨투어 내부 평균 명도 - 주변 ring 평균 명도 ≥ contrast_th
"""

import time

import cv2
import numpy as np

# DETECTING 오버레이 두께/폰트 (mainViewer.py 에서 import)
MARKER_THICKNESS = 3   # GC 컨투어 두께
ROI_THICKNESS = 5      # ROI 사각형 두께
FONT_SCALE = 5.0       # Count / ROI 라벨 폰트 크기
FONT_THICKNESS = 5     # 폰트 두께


class GobletDetector:
    def detect(self, image: np.ndarray, params: dict) -> tuple[list, int, float]:
        """술잔세포 검출. 반환: (contours, count, elapsed_ms)."""
        start_time = time.time()
        if image is None:
            return [], 0, 0.0

        h, w = image.shape[:2]
        reference_length = (w + h) / 2.0

        rx, ry = int(params.get("ROI_X", 0)), int(params.get("ROI_Y", 0))
        rw, rh = int(params.get("ROI_W", 0)), int(params.get("ROI_H", 0))

        contrast_th    = params.get("Contrast", 5.0)
        min_size_px    = ((params.get("MinSize", 0.3) / 100.0) * reference_length) ** 2
        max_size_px    = ((params.get("MaxSize", 1.0) / 100.0) * reference_length) ** 2
        block_size_px  = int((params.get("BlockSize", 1.0) / 100.0) * reference_length)
        if block_size_px < 3:
            block_size_px = 3
        if block_size_px % 2 == 0:
            block_size_px += 1
        adaptive_c      = int(params.get("AdaptiveC", 0.5))
        sensitivity     = params.get("Sensitivity", 45.0) / 100.0
        min_circularity = params.get("Circularity", 20.0) / 100.0

        final_contours = []
        if rw > 0 and rh > 0:
            roi_gray = image[ry:ry + rh, rx:rx + rw]
            binary_roi = cv2.adaptiveThreshold(
                cv2.medianBlur(roi_gray, 5), 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                block_size_px, adaptive_c,
            )
            contours = self._watershed_segmentation(binary_roi, roi_gray, sensitivity)

            for contour in contours:
                area = cv2.contourArea(contour)
                if not (min_size_px <= area <= max_size_px):
                    continue
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                if circularity < min_circularity:
                    continue

                # Contrast check: 내부 평균 vs 주변 dilate ring 평균
                mask = np.zeros(roi_gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_inside = cv2.mean(roi_gray, mask=mask)[0]
                bg_mask = cv2.subtract(cv2.dilate(mask, np.ones((10, 10), np.uint8)), mask)
                if np.sum(bg_mask) == 0:
                    continue
                if (mean_inside - cv2.mean(roi_gray, mask=bg_mask)[0]) < contrast_th:
                    continue

                final_contours.append(contour + (rx, ry))  # ROI → 전체 이미지 좌표계로 평행이동

        elapsed_ms = (time.time() - start_time) * 1000.0
        return final_contours, len(final_contours), elapsed_ms

    @staticmethod
    def _watershed_segmentation(binary_img: np.ndarray, gray_img: np.ndarray,
                                sensitivity: float) -> list:
        """Distance transform + watershed 으로 군집된 점들을 개별 컨투어로 분리."""
        sure_bg = cv2.dilate(binary_img, np.ones((3, 3), np.uint8), iterations=2)
        dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
            dist_transform, sensitivity * dist_transform.max(), 255, 0
        )
        unknown = cv2.subtract(sure_bg, np.uint8(sure_fg))
        _, markers = cv2.connectedComponents(np.uint8(sure_fg))
        markers += 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), markers)

        res = []
        for label in np.unique(markers):
            if label <= 1:
                continue
            mask = np.zeros(gray_img.shape, dtype="uint8")
            mask[markers == label] = 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            res.extend(cnts)
        return res
