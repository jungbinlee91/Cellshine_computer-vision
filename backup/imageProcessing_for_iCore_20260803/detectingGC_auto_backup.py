import time

import cv2
import numpy as np

# DETECTING 오버레이 두께/폰트
MARKER_THICKNESS = 3
ROI_THICKNESS = 5
FONT_SCALE = 5.0
FONT_THICKNESS = 5


class GobletDetector_Auto:
    """술잔세포 검출기."""

    # 매번 새로 만들 필요가 없는 커널을 한 번만 생성
    _WS_KERNEL = np.ones((3, 3), dtype=np.uint8)
    _CONTRAST_KERNEL = np.ones((10, 10), dtype=np.uint8)

    def detect_auto(
        self,
        image: np.ndarray,
        params: dict,
    ) -> tuple[list[np.ndarray], int, float]:
        """술잔세포 검출.

        반환:
            contours: 전체 이미지 좌표계의 최종 컨투어
            count: 검출된 술잔세포 개수
            elapsed_ms: 전체 처리시간(ms)
        """
        start_time = time.perf_counter()

        if image is None:
            return [], 0, 0.0

        h, w = image.shape[:2]
        reference_length = (w + h) / 2.0

        # ROI
        rx = int(params.get("ROI_X", 0))
        ry = int(params.get("ROI_Y", 0))
        rw = int(params.get("ROI_W", 0))
        rh = int(params.get("ROI_H", 0))

        # 검출 파라미터
        contrast_th = float(params.get("Contrast", 5.0))

        min_size_px = (
            (float(params.get("MinSize", 0.3)) / 100.0)
            * reference_length
        ) ** 2

        max_size_px = (
            (float(params.get("MaxSize", 1.0)) / 100.0)
            * reference_length
        ) ** 2

        block_size_px = int(
            (float(params.get("BlockSize", 1.0)) / 100.0)
            * reference_length
        )
        block_size_px = max(block_size_px, 3)

        if block_size_px % 2 == 0:
            block_size_px += 1

        # 기존 코드와 동일하게 int 변환
        adaptive_c = int(params.get("AdaptiveC", 0.5))
        sensitivity = float(params.get("Sensitivity", 45.0)) / 100.0
        min_circularity = float(params.get("Circularity", 20.0)) / 100.0

        if rw <= 0 or rh <= 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return [], 0, elapsed_ms

        # 1. ROI 추출
        roi_gray = image[ry:ry + rh, rx:rx + rw]

        if roi_gray.size == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return [], 0, elapsed_ms

        # 2. Median blur
        blurred_roi = cv2.medianBlur(roi_gray, 5)

        # 3. Adaptive threshold
        binary_roi = cv2.adaptiveThreshold(
            blurred_roi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size_px,
            adaptive_c,
        )

        # 4. Watershed
        contours = self._watershed_segmentation(
            binary_roi,
            roi_gray,
            sensitivity,
        )

        final_contours: list[np.ndarray] = []

        # 반복문 내부의 상수와 속성을 지역 변수로 저장
        pi4 = 4.0 * np.pi
        roi_height, roi_width = roi_gray.shape[:2]

        for contour in contours:
            # 5-1. 면적 필터
            area = cv2.contourArea(contour)

            if area < min_size_px or area > max_size_px:
                continue

            # 5-2. 원형도 필터
            perimeter = cv2.arcLength(contour, True)

            if perimeter <= 0.0:
                continue

            circularity = (pi4 * area) / (perimeter * perimeter)

            if circularity < min_circularity:
                continue

            # 5-3. 내부-주변 명암 차이 검사
            if not self._passes_contrast_check(
                roi_gray=roi_gray,
                contour=contour,
                contrast_th=contrast_th,
                roi_width=roi_width,
                roi_height=roi_height,
            ):
                continue

            # ROI 좌표계에서 전체 이미지 좌표계로 이동
            final_contours.append(
                contour + np.array([rx, ry], dtype=contour.dtype)
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return final_contours, len(final_contours), elapsed_ms

    @classmethod
    def _passes_contrast_check(
        cls,
        roi_gray: np.ndarray,
        contour: np.ndarray,
        contrast_th: float,
        roi_width: int,
        roi_height: int,
    ) -> bool:
        """컨투어 내부와 주변 ring 영역의 평균 명도 차이를 검사.

        기존 코드와 같은 검사를 수행하지만 ROI 전체가 아니라
        컨투어 주변의 작은 영역만 계산한다.
        """
        x, y, width, height = cv2.boundingRect(contour)

        # 10×10 dilation이 컨투어 밖으로 확장되는 범위를 포함
        margin = 5

        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(roi_width, x + width + margin)
        y1 = min(roi_height, y + height + margin)

        local_gray = roi_gray[y0:y1, x0:x1]

        if local_gray.size == 0:
            return False

        # 컨투어를 지역 좌표계로 이동
        offset = np.array([x0, y0], dtype=contour.dtype)
        local_contour = contour - offset

        # 객체 내부 마스크
        local_mask = np.zeros(local_gray.shape, dtype=np.uint8)
        cv2.drawContours(
            local_mask,
            [local_contour],
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED,
        )

        mean_inside = cv2.mean(local_gray, mask=local_mask)[0]

        # 객체 주변 ring 마스크
        dilated_mask = cv2.dilate(
            local_mask,
            cls._CONTRAST_KERNEL,
            iterations=1,
        )
        ring_mask = cv2.subtract(dilated_mask, local_mask)

        if cv2.countNonZero(ring_mask) == 0:
            return False

        mean_background = cv2.mean(
            local_gray,
            mask=ring_mask,
        )[0]

        return (mean_inside - mean_background) >= contrast_th

    @classmethod
    def _watershed_segmentation(
        cls,
        binary_img: np.ndarray,
        gray_img: np.ndarray,
        sensitivity: float,
    ) -> list[np.ndarray]:
        """Distance transform + Watershed로 군집된 객체를 분리."""
        # 확실한 배경
        sure_bg = cv2.dilate(
            binary_img,
            cls._WS_KERNEL,
            iterations=2,
        )

        # 객체 중심 거리 계산
        dist_transform = cv2.distanceTransform(
            binary_img,
            cv2.DIST_L2,
            5,
        )

        # np.max 대신 OpenCV 함수로 최댓값 계산
        _, max_distance, _, _ = cv2.minMaxLoc(dist_transform)

        _, sure_fg = cv2.threshold(
            dist_transform,
            sensitivity * max_distance,
            255,
            cv2.THRESH_BINARY,
        )

        sure_fg = np.uint8(sure_fg)

        # 배경과 전경 사이의 불확실 영역
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 각 전경 영역에 marker 번호 부여
        number_of_labels, markers = cv2.connectedComponents(sure_fg)

        markers += 1
        markers[unknown == 255] = 0

        # Watershed는 3채널 영상 필요
        watershed_image = cv2.cvtColor(
            gray_img,
            cv2.COLOR_GRAY2BGR,
        )

        cv2.watershed(watershed_image, markers)

        contours: list[np.ndarray] = []

        # np.unique(markers)를 다시 계산하지 않고
        # connectedComponents가 만든 label 범위를 직접 순회
        for label in range(2, number_of_labels + 1):
            # NumPy 마스킹보다 OpenCV 내부 연산을 사용
            label_mask = cv2.compare(
                markers,
                label,
                cv2.CMP_EQ,
            )

            if cv2.countNonZero(label_mask) == 0:
                continue

            label_contours, _ = cv2.findContours(
                label_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            contours.extend(label_contours)

        return contours