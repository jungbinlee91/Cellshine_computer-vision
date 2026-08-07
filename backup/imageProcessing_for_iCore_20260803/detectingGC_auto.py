import time

import cv2
import numpy as np

# DETECTING 오버레이 설정
MARKER_THICKNESS = 3
ROI_THICKNESS = 5
FONT_SCALE = 5.0
FONT_THICKNESS = 5


class GobletDetector_Auto:

    def detect_auto(
        self,
        image: np.ndarray,
        params: dict,
    ) -> tuple[list, int, float]:
        """
        술잔세포 검출.

        Pipeline
        --------
        1. ROI
        2. Gaussian denoise
        3. Local contrast normalization
        4. Multi-scale DoG
        5. DoG local peak 검출
        6. Center-ring contrast 검증
        7. Scale-adaptive NMS
        8. 원영상 특성을 이용한 contour 생성

        Returns
        -------
        contours:
            전체 이미지 좌표계 contour list

        count:
            GC 개수

        elapsed_ms:
            처리 시간 [ms]
        """

        start_time = time.time()

        if image is None:
            return [], 0, 0.0

        # ==========================================================
        # 0. Grayscale
        # ==========================================================

        if image.ndim == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        if image.ndim != 2:
            return [], 0, 0.0

        full_h, full_w = image.shape[:2]

        # ==========================================================
        # 1. Parameters
        # ==========================================================

        rx = int(params.get("ROI_X", 0))
        ry = int(params.get("ROI_Y", 0))
        rw = int(params.get("ROI_W", 0))
        rh = int(params.get("ROI_H", 0))

        # 예상 GC 직경 [pixel]
        min_diameter = float(
            params.get("MinDiameter", 8.0)
        )

        max_diameter = float(
            params.get("MaxDiameter", 30.0)
        )

        # 0 ~ 100
        sensitivity = float(
            params.get("Sensitivity", 60.0)
        ) / 100.0

        # local-normalized intensity 기준
        min_center_ring_contrast = float(
            params.get(
                "CenterRingContrast",
                0.30,
            )
        )

        # DoG scale 개수
        num_scales = max(
            int(params.get("NumScales", 6)),
            3,
        )

        # 같은 GC 안에서 여러 peak가 생기는 것을 제거하는 정도
        peak_distance_ratio = float(
            params.get(
                "PeakDistanceRatio",
                0.55,
            )
        )

        sensitivity = float(
            np.clip(
                sensitivity,
                0.0,
                1.0,
            )
        )

        min_diameter = max(
            min_diameter,
            2.0,
        )

        max_diameter = max(
            max_diameter,
            min_diameter + 1.0,
        )

        peak_distance_ratio = float(
            np.clip(
                peak_distance_ratio,
                0.2,
                1.0,
            )
        )

        # ==========================================================
        # 2. ROI
        # ==========================================================

        if rw <= 0 or rh <= 0:
            elapsed_ms = (
                time.time() - start_time
            ) * 1000.0

            return [], 0, elapsed_ms

        rx = max(0, rx)
        ry = max(0, ry)

        rw = min(
            rw,
            full_w - rx,
        )

        rh = min(
            rh,
            full_h - ry,
        )

        if rw <= 0 or rh <= 0:
            elapsed_ms = (
                time.time() - start_time
            ) * 1000.0

            return [], 0, elapsed_ms

        roi_gray = image[
            ry:ry + rh,
            rx:rx + rw
        ]

        img = roi_gray.astype(
            np.float32
        )

        # ==========================================================
        # 3. 약한 denoise
        # ==========================================================

        denoise_sigma = float(
            np.clip(
                min_diameter / 8.0,
                0.8,
                1.5,
            )
        )

        smooth = cv2.GaussianBlur(
            img,
            (0, 0),
            sigmaX=denoise_sigma,
            sigmaY=denoise_sigma,
        )

        # ==========================================================
        # 4. Local contrast normalization
        # ==========================================================

        background_sigma = max(
            max_diameter * 1.5,
            10.0,
        )

        local_mean = cv2.GaussianBlur(
            smooth,
            (0, 0),
            sigmaX=background_sigma,
            sigmaY=background_sigma,
        )

        local_mean_sq = cv2.GaussianBlur(
            smooth * smooth,
            (0, 0),
            sigmaX=background_sigma,
            sigmaY=background_sigma,
        )

        local_var = np.maximum(
            local_mean_sq
            - local_mean * local_mean,
            0.0,
        )

        local_std = np.sqrt(
            local_var
        )

        # ----------------------------------------------------------
        # Noise floor 자동 추정
        # ----------------------------------------------------------

        valid_std = local_std[
            local_std > 1e-6
        ]

        if valid_std.size > 0:
            noise_floor = float(
                np.percentile(
                    valid_std,
                    20,
                )
            )
        else:
            noise_floor = 1.0

        global_std = float(
            np.std(smooth)
        )

        noise_floor = max(
            noise_floor,
            global_std * 0.10,
            1e-3,
        )

        normalized = (
            smooth - local_mean
        ) / np.maximum(
            local_std,
            noise_floor,
        )

        # 5. Multi-scale DoG + 3D scale-space NMS
        #
        # 같은 구조가 여러 scale에서 반복 검출되는 것을 막으면서
        # 각 GC가 가장 잘 보이는 scale의 peak만 남긴다.
        # ==========================================================

        sigma_min = max(
            min_diameter / 3.0,
            0.8,
        )

        sigma_max = max(
            max_diameter / 3.0,
            sigma_min + 0.1,
        )

        sigmas = np.geomspace(
            sigma_min,
            sigma_max,
            num_scales,
        )

        dog_ratio = 1.6

        threshold_sigma = (
            4.5 - 3.0 * sensitivity
        )

        score_maps = []

        # ----------------------------------------------------------
        # scale별 DoG + robust normalization
        # ----------------------------------------------------------

        for sigma in sigmas:

            sigma = float(sigma)

            small = cv2.GaussianBlur(
                normalized,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
            )

            large = cv2.GaussianBlur(
                normalized,
                (0, 0),
                sigmaX=sigma * dog_ratio,
                sigmaY=sigma * dog_ratio,
            )

            dog = small - large

            median_response = float(
                np.median(dog)
            )

            mad = float(
                np.median(
                    np.abs(
                        dog - median_response
                    )
                )
            )

            robust_sigma = (
                1.4826 * mad
            )

            if robust_sigma < 1e-6:
                robust_sigma = float(
                    np.std(dog)
                )

            robust_sigma = max(
                robust_sigma,
                1e-6,
            )

            # scale마다 response 크기가 다르므로
            # robust z-score로 정규화
            score = (
                dog - median_response
            ) / robust_sigma

            score_maps.append(
                score.astype(
                    np.float32,
                    copy=False,
                )
            )

        score_stack = np.stack(
            score_maps,
            axis=0,
        )

        # ==========================================================
        # 6. 3D local maximum
        #
        # x-y 방향뿐 아니라
        # 앞/뒤 scale에서도 가장 강한 peak만 인정
        # ==========================================================

        peak_candidates = []

        kernel = np.ones(
            (3, 3),
            dtype=np.uint8,
        )

        for scale_index in range(
            len(sigmas)
        ):

            score = score_stack[
                scale_index
            ]

            # 현재 scale에서 spatial maximum
            spatial_max = cv2.dilate(
                score,
                kernel,
            )

            peak_mask = (
                (score >= spatial_max - 1e-6)
                &
                (score > threshold_sigma)
            )

            # ------------------------------------------------------
            # 이전 scale의 주변 3x3보다도 강해야 함
            # ------------------------------------------------------

            if scale_index > 0:

                prev_max = cv2.dilate(
                    score_stack[
                        scale_index - 1
                    ],
                    kernel,
                )

                peak_mask &= (
                    score >= prev_max - 1e-6
                )

            # ------------------------------------------------------
            # 다음 scale의 주변 3x3보다도 강해야 함
            # ------------------------------------------------------

            if (
                scale_index
                < len(sigmas) - 1
            ):

                next_max = cv2.dilate(
                    score_stack[
                        scale_index + 1
                    ],
                    kernel,
                )

                peak_mask &= (
                    score >= next_max - 1e-6
                )

            ys, xs = np.where(
                peak_mask
            )

            for cy, cx in zip(
                ys,
                xs,
            ):
                # ------------------------------------------------------
                # Scale persistence 검사
                #
                # 실제 blob이라면 가장 강한 scale뿐 아니라
                # 인접 scale에서도 어느 정도 반응이 유지되어야 함
                # ------------------------------------------------------

                current_score = float(
                    score[cy, cx]
                )

                support_scores = []

                # 이전 scale
                if scale_index > 0:

                    y0 = max(cy - 1, 0)
                    y1 = min(cy + 2, score.shape[0])

                    x0 = max(cx - 1, 0)
                    x1 = min(cx + 2, score.shape[1])

                    prev_support = float(
                        np.max(
                            score_stack[
                                scale_index - 1,
                                y0:y1,
                                x0:x1,
                            ]
                        )
                    )

                    support_scores.append(
                        prev_support
                    )

                # 다음 scale
                if scale_index < len(sigmas) - 1:

                    y0 = max(cy - 1, 0)
                    y1 = min(cy + 2, score.shape[0])

                    x0 = max(cx - 1, 0)
                    x1 = min(cx + 2, score.shape[1])

                    next_support = float(
                        np.max(
                            score_stack[
                                scale_index + 1,
                                y0:y1,
                                x0:x1,
                            ]
                        )
                    )

                    support_scores.append(
                        next_support
                    )

                # 인접 scale 중 적어도 하나에서
                # 현재 peak의 40% 이상 반응이 남아 있어야 함
                if support_scores:

                    persistence = (
                        max(support_scores)
                        / max(current_score, 1e-6)
                    )

                    if persistence < 0.40:
                        continue

                peak_candidates.append(
                    {
                        "x": int(cx),
                        "y": int(cy),
                        "scale_index":
                            scale_index,
                        "score": float(
                            score[cy, cx]
                        ),
                    }
                )

        # 필요 없어진 큰 배열 해제
        del score_stack
        del score_maps

        if not peak_candidates:

            elapsed_ms = (
                time.time()
                - start_time
            ) * 1000.0

            return [], 0, elapsed_ms

        # 강한 peak부터 후속 검사
        peak_candidates.sort(
            key=lambda p: p["score"],
            reverse=True,
        )

        # ==========================================================
        # 8. Center-ring 검사 + scale-adaptive NMS
        # ==========================================================

        accepted = []

        for candidate in peak_candidates:

            cx = candidate["x"]
            cy = candidate["y"]

            scale_index = (
                candidate["scale_index"]
            )

            estimated_diameter = float(
                3.0
                * sigmas[scale_index]
            )

            estimated_diameter = float(
                np.clip(
                    estimated_diameter,
                    min_diameter,
                    max_diameter,
                )
            )

            # ------------------------------------------------------
            # 실제로 주변보다 밝은가?
            # ------------------------------------------------------

            contrast_result = (
                self._center_ring_contrast(
                    normalized,
                    cx,
                    cy,
                    estimated_diameter,
                )
            )

            if contrast_result is None:
                continue

            (
                center_intensity,
                ring_intensity,
                center_ring_contrast,
            ) = contrast_result

            if (
                center_ring_contrast
                < min_center_ring_contrast
            ):
                continue

            # ------------------------------------------------------
            # Scale-adaptive NMS
            #
            # 같은 큰 GC 내부에서 여러 peak가 생기는 것은 제거
            # 서로 붙어 있어도 중심이 충분히 떨어져 있으면 보존
            # ------------------------------------------------------

            duplicate = False

            for existing in accepted:

                dx = (
                    cx
                    - existing["x"]
                )

                dy = (
                    cy
                    - existing["y"]
                )

                distance = np.hypot(
                    dx,
                    dy,
                )

                required_distance = (
                    peak_distance_ratio
                    * min(
                        estimated_diameter,
                        existing["diameter"],
                    )
                )

                required_distance = max(
                    required_distance,
                    2.0,
                )

                if (
                    distance
                    < required_distance
                ):
                    duplicate = True
                    break

            if duplicate:
                continue

            accepted.append(
                {
                    "x": cx,
                    "y": cy,
                    "diameter":
                        estimated_diameter,
                    "center":
                        center_intensity,
                    "ring":
                        ring_intensity,
                    "contrast":
                        center_ring_contrast,
                    "score": candidate["score"],
                }
            )

        # ==========================================================
        # 9. 각각의 GC 중심으로 실제 contour 생성
        # ==========================================================

        final_contours = []

        for i, cell in enumerate(
            accepted
        ):

            contour = (
                self._build_cell_contour(
                    normalized,
                    cell,
                    accepted,
                    i,
                )
            )

            # segmentation이 불안정한 경우
            # 검출 자체를 버리지 않고 예상 크기의 contour 사용
            if contour is None:
                continue

            # ==========================================================
            # 실제 만들어진 contour 크기 검사
            # ==========================================================

            area = cv2.contourArea(contour)

            if area <= 0:
                continue

            actual_diameter = np.sqrt(
                4.0 * area / np.pi
            )

            # 실제 blob의 상당직경이 너무 작으면 노이즈로 판단
            min_actual_diameter = (
                min_diameter * 0.75
            )

            if actual_diameter < min_actual_diameter:
                continue

            # ROI → 전체 영상 좌표
            global_contour = (
                contour.copy()
            )

            global_contour[
                :, 0, 0
            ] += rx

            global_contour[
                :, 0, 1
            ] += ry

            final_contours.append(
                global_contour
            )

        # ==========================================================
        # 10. Result
        # ==========================================================

        elapsed_ms = (
            time.time()
            - start_time
        ) * 1000.0

        return (
            final_contours,
            len(final_contours),
            elapsed_ms,
        )

    # ==============================================================
    # Center-ring contrast
    # ==============================================================

    @staticmethod
    def _center_ring_contrast(
        normalized: np.ndarray,
        cx: int,
        cy: int,
        diameter: float,
    ):

        radius = (
            diameter / 2.0
        )

        center_radius = max(
            radius * 0.45,
            1.0,
        )

        outer_radius = max(
            radius * 1.5,
            2.0,
        )

        h, w = normalized.shape

        x0 = max(
            0,
            int(cx - outer_radius - 1),
        )

        x1 = min(
            w,
            int(cx + outer_radius + 2),
        )

        y0 = max(
            0,
            int(cy - outer_radius - 1),
        )

        y1 = min(
            h,
            int(cy + outer_radius + 2),
        )

        if x1 <= x0 or y1 <= y0:
            return None

        patch = normalized[
            y0:y1,
            x0:x1
        ]

        yy, xx = np.ogrid[
            y0:y1,
            x0:x1
        ]

        dist_sq = (
            (xx - cx) ** 2
            + (yy - cy) ** 2
        )

        center_region = (
            dist_sq
            <= center_radius ** 2
        )

        ring_region = (
            (
                dist_sq
                >= (radius * 0.8) ** 2
            )
            &
            (
                dist_sq
                <= (radius * 1.5) ** 2
            )
        )

        if (
            not np.any(
                center_region
            )
            or
            not np.any(
                ring_region
            )
        ):
            return None

        center_intensity = float(
            np.mean(
                patch[
                    center_region
                ]
            )
        )

        ring_intensity = float(
            np.percentile(
                patch[ring_region],
                30,
            )
        )

        contrast = (
            center_intensity
            - ring_intensity
        )

        return (
            center_intensity,
            ring_intensity,
            contrast,
        )

    # ==============================================================
    # 실제 GC contour 생성
    # ==============================================================

    @staticmethod
    def _build_cell_contour(
        normalized: np.ndarray,
        cell: dict,
        cells: list,
        cell_index: int,
    ):

        cx = int(cell["x"])
        cy = int(cell["y"])

        diameter = float(
            cell["diameter"]
        )

        radius = (
            diameter / 2.0
        )

        # 예상 GC보다 약간 넓은 범위만 탐색
        search_radius = max(
            radius * 1.15,
            2.0,
        )

        h, w = normalized.shape

        x0 = max(
            0,
            int(cx - search_radius - 2),
        )

        x1 = min(
            w,
            int(cx + search_radius + 3),
        )

        y0 = max(
            0,
            int(cy - search_radius - 2),
        )

        y1 = min(
            h,
            int(cy + search_radius + 3),
        )

        if x1 <= x0 or y1 <= y0:
            return None

        patch = normalized[
            y0:y1,
            x0:x1
        ]

        yy, xx = np.ogrid[
            y0:y1,
            x0:x1
        ]

        dist_current_sq = (
            (xx - cx) ** 2
            + (yy - cy) ** 2
        )

        # ----------------------------------------------------------
        # 세포 밝기와 주변 밝기를 이용해서
        # 해당 세포용 threshold 자동 결정
        # ----------------------------------------------------------

        contrast = max(
            float(cell["contrast"]),
            0.0,
        )

        threshold_level = (
            float(cell["ring"])
            + max(
                0.08,
                contrast * 0.25,
            )
        )

        foreground = (
            patch
            > threshold_level
        ).astype(np.uint8)

        # GC 예상 영역 밖 제거
        allowed = (
            dist_current_sq
            <= search_radius ** 2
        )

        # ----------------------------------------------------------
        # 인접 GC와 붙어있는 경우
        # 두 peak 사이를 자동으로 나눔
        #
        # watershed가 아니라
        # 각 GC 중심과의 상대거리로 분할
        # ----------------------------------------------------------

        current_radius_sq = max(
            radius ** 2,
            1.0,
        )

        normalized_current_distance = (
            dist_current_sq
            / current_radius_sq
        )

        for j, other in enumerate(
            cells
        ):

            if j == cell_index:
                continue

            ox = int(other["x"])
            oy = int(other["y"])

            other_radius = (
                float(
                    other["diameter"]
                )
                / 2.0
            )

            center_distance_sq = (
                (cx - ox) ** 2
                + (cy - oy) ** 2
            )

            # 충분히 멀리 있는 GC는 계산할 필요 없음
            interaction_distance = (
                search_radius
                + other_radius * 1.2
            )

            if (
                center_distance_sq
                > interaction_distance ** 2
            ):
                continue

            dist_other_sq = (
                (xx - ox) ** 2
                + (yy - oy) ** 2
            )

            other_radius_sq = max(
                other_radius ** 2,
                1.0,
            )

            normalized_other_distance = (
                dist_other_sq
                / other_radius_sq
            )

            allowed &= (
                normalized_current_distance
                <= normalized_other_distance
            )

        # 작은 hole이나 끊김 완화
        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3),
            ),
        )

        foreground[
            ~allowed
        ] = 0

        # ----------------------------------------------------------
        # 중심과 연결된 영역 선택
        # ----------------------------------------------------------

        num_labels, labels = (
            cv2.connectedComponents(
                foreground,
                connectivity=8,
            )
        )

        if num_labels <= 1:
            return None

        seed_radius = max(
            radius * 0.25,
            1.5,
        )

        seed_region = (
            dist_current_sq
            <= seed_radius ** 2
        )

        best_label = 0
        best_overlap = 0

        for label in range(
            1,
            num_labels,
        ):

            overlap = np.count_nonzero(
                (labels == label)
                & seed_region
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label

        # 중심부와 직접 겹치는 component가 없다면
        # 가장 가까운 component 선택
        if best_label == 0:

            fg_y, fg_x = np.where(
                labels > 0
            )

            if fg_x.size == 0:
                return None

            global_x = (
                fg_x + x0
            )

            global_y = (
                fg_y + y0
            )

            distances = (
                (global_x - cx) ** 2
                + (global_y - cy) ** 2
            )

            nearest = int(
                np.argmin(
                    distances
                )
            )

            best_label = int(
                labels[
                    fg_y[nearest],
                    fg_x[nearest]
                ]
            )

        cell_mask = (
            labels
            == best_label
        ).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            cell_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        # patch 좌표 → ROI 좌표
        contour = contour.copy()

        contour[:, 0, 0] += x0
        contour[:, 0, 1] += y0

        return contour

    # ==============================================================
    # contour segmentation 실패 시 fallback
    # ==============================================================

    @staticmethod
    def _fallback_circle(
        cx: int,
        cy: int,
        diameter: float,
        width: int,
        height: int,
    ):

        radius = max(
            int(round(
                diameter / 2.0
            )),
            1,
        )

        points = cv2.ellipse2Poly(
            (cx, cy),
            (radius, radius),
            0,
            0,
            360,
            15,
        )

        if points.size == 0:
            return None

        points[:, 0] = np.clip(
            points[:, 0],
            0,
            width - 1,
        )

        points[:, 1] = np.clip(
            points[:, 1],
            0,
            height - 1,
        )

        return points.reshape(
            -1,
            1,
            2,
        ).astype(np.int32)