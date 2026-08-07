import time

import cv2
import numpy as np

# DETECTING 오버레이 설정
MARKER_THICKNESS = 3
ROI_THICKNESS = 5
FONT_SCALE = 5.0
FONT_THICKNESS = 5


class GobletDetector:

    def detect(
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
            params.get("MinDiameter", 20.0)
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

        abs_threshold = float(
            params.get(
                "AbsThreshold",
                140,
            )
        )

        bright_rescue_intensity = float(
            params.get(
                "BrightRescueIntensity",
                170.0,
            )
        )

        bright_rescue_response_ratio = float(
            params.get(
                "BrightRescueResponseRatio",
                0.65,
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

        # ==========================================================
        # 5. Multi-scale DoG
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

        best_response = np.full(
            normalized.shape,
            -np.inf,
            dtype=np.float32,
        )

        best_scale = np.zeros(
            normalized.shape,
            dtype=np.uint8,
        )

        dog_responses = []

        for i, sigma in enumerate(sigmas):

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

            dog = (
                small - large
            )

            dog_responses.append(dog)

            update = (
                dog > best_response
            )

            best_response[update] = (
                dog[update]
            )

            best_scale[update] = i

        # ==========================================================
        # 6. 자동 DoG threshold
        # ==========================================================

        median_response = float(
            np.median(
                best_response
            )
        )

        mad = float(
            np.median(
                np.abs(
                    best_response
                    - median_response
                )
            )
        )

        robust_sigma = (
            1.4826 * mad
        )

        if robust_sigma < 1e-6:
            robust_sigma = float(
                np.std(
                    best_response
                )
            )

        threshold_sigma = (
            4.5
            - 3.0 * sensitivity
        )

        response_threshold = (
            median_response
            + threshold_sigma
            * robust_sigma
        )

        response_threshold = max(
            response_threshold,
            0.0,
        )

        # ==========================================================
        # 7. DoG local peak 검출
        #
        # 기존 best_response peak는 그대로 보존하고,
        # 각 scale에서만 보이는 peak도 추가로 회수한다.
        # ==========================================================

        kernel = np.ones(
            (3, 3),
            dtype=np.uint8,
        )

        # ----------------------------------------------------------
        # 기존 방식의 peak
        # ----------------------------------------------------------

        local_max = cv2.dilate(
            best_response,
            kernel,
        )

        main_peak_mask = (
            (best_response >= local_max - 1e-6)
            &
            (best_response > response_threshold)
        )

        # 후보 정보를 저장할 map
        candidate_response_map = np.full(
            normalized.shape,
            -np.inf,
            dtype=np.float32,
        )

        candidate_scale_map = np.zeros(
            normalized.shape,
            dtype=np.uint8,
        )

        candidate_response_map[
            main_peak_mask
        ] = best_response[
            main_peak_mask
        ]

        candidate_scale_map[
            main_peak_mask
        ] = best_scale[
            main_peak_mask
        ]

        # ----------------------------------------------------------
        # Bright-rescue용 threshold
        # ----------------------------------------------------------

        rescue_response_threshold = (
            response_threshold
            * bright_rescue_response_ratio
        )

        # ----------------------------------------------------------
        # 각 DoG scale에서 개별적으로 local peak 검색
        # ----------------------------------------------------------

        for scale_index, dog in enumerate(
            dog_responses
        ):

            scale_local_max = cv2.dilate(
                dog,
                kernel,
            )

            scale_peak_mask = (
                (dog >= scale_local_max - 1e-6)
                &
                (dog > rescue_response_threshold)
            )

            # 기존 후보가 없는 위치에
            # scale-specific peak를 추가
            update = (
                scale_peak_mask
                &
                (
                    dog
                    > candidate_response_map
                )
            )

            candidate_response_map[
                update
            ] = dog[
                update
            ]

            candidate_scale_map[
                update
            ] = scale_index


        peak_mask = np.isfinite(
            candidate_response_map
        )

        peak_ys, peak_xs = np.where(
            peak_mask
        )

        if peak_xs.size == 0:

            elapsed_ms = (
                time.time() - start_time
            ) * 1000.0

            return [], 0, elapsed_ms


        peak_values = (
            candidate_response_map[
                peak_ys,
                peak_xs
            ]
        )

        peak_scales = (
            candidate_scale_map[
                peak_ys,
                peak_xs
            ]
        )

        # 강한 후보부터 검사
        order = np.argsort(
            peak_values
        )[::-1]

        # ==========================================================
        # 8. Center-ring 검사 + scale-adaptive NMS
        # ==========================================================

        accepted = []

        for index in order:

            cx = int(
                peak_xs[index]
            )

            cy = int(
                peak_ys[index]
            )

            scale_index = int(
                peak_scales[index]
            )

            candidate_response = float(
                peak_values[index]
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
                    smooth,
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
                robust_contrast,
                absolute_center_intensity,
            ) = contrast_result

            if (
                center_ring_contrast
                < min_center_ring_contrast
            ):
                continue


            # 지금까지 잘 작동하던 기존 검출 조건
            normal_pass = (
                candidate_response
                > response_threshold
                and
                robust_contrast >= 0.8
                and
                absolute_center_intensity >= abs_threshold
            )


            # 밝기는 확실히 강하지만
            # DoG/robust contrast가 약한 GC를 구제
            bright_rescue_pass = (
                candidate_response
                > rescue_response_threshold
                and
                absolute_center_intensity
                >= bright_rescue_intensity
            )


            if not (
                normal_pass
                or bright_rescue_pass
            ):
                continue

            # # 중심이 주변 8방향 중 최소 6방향보다
            # # 밝은 경우에만 blob으로 인정
            # if directional_support < 1: # 0.75
            #     continue

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
                    "response":
                        candidate_response,
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

            # 실제로 세포 크기에 해당하는 밝은 영역이 존재하는지 검사
            area = cv2.contourArea(contour)

            if area <= 0:
                continue

            actual_diameter = np.sqrt(
                4.0 * area / np.pi
            )

            # 실제 만들어진 blob 자체가 너무 작으면 제거
            min_actual_diameter = (
                min_diameter * 1
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
        smooth: np.ndarray,
        cx: int,
        cy: int,
        diameter: float,
    ):

        radius = diameter / 2.0

        center_radius = max(
            radius * 0.45,
            1.0,
        )

        # robust ring까지 보기 위해 조금 넓힘
        outer_radius = max(
            radius * 1.7,
            2.0,
        )

        h, w = normalized.shape

        x0 = max(0, int(cx - outer_radius - 1))
        x1 = min(w, int(cx + outer_radius + 2))
        y0 = max(0, int(cy - outer_radius - 1))
        y1 = min(h, int(cy + outer_radius + 2))

        if x1 <= x0 or y1 <= y0:
            return None

        patch = normalized[y0:y1, x0:x1]

        signal_patch = smooth[y0:y1,x0:x1]

        radius = diameter / 2.0

        center_radius = max(
            radius * 0.45,
            1.0,
        )

        outer_radius = max(
            radius * 1.7,
            2.0,
        )

        h, w = normalized.shape

        x0 = max(0, int(cx - outer_radius - 1))
        x1 = min(w, int(cx + outer_radius + 2))
        y0 = max(0, int(cy - outer_radius - 1))
        y1 = min(h, int(cy + outer_radius + 2))

        # normalized 영상
        patch = normalized[
            y0:y1,
            x0:x1
        ]

        # 실제 밝기를 측정할 영상
        signal_patch = smooth[
            y0:y1,
            x0:x1
        ]

        # 각 pixel의 실제 영상 좌표
        yy, xx = np.ogrid[
            y0:y1,
            x0:x1
        ]

        # 후보 중심(cx, cy)으로부터 각 pixel까지의 거리²
        dist_sq = (
            (xx - cx) ** 2
            + (yy - cy) ** 2
        )

        # 바로 이게 center_region
        center_region = (
            dist_sq <= center_radius ** 2
        )

        absolute_center_intensity = float(np.percentile(signal_patch[center_region],70,))

        yy, xx = np.ogrid[y0:y1, x0:x1]

        dist_sq = (
            (xx - cx) ** 2
            + (yy - cy) ** 2
        )

        center_region = (
            dist_sq <= center_radius ** 2
        )

        # 기존 ring
        ring_region = (
            (dist_sq >= (radius * 0.8) ** 2)
            &
            (dist_sq <= (radius * 1.5) ** 2)
        )

        # 혈관의 영향을 덜 받는 바깥쪽 ring
        robust_ring_region = (
            (dist_sq >= (radius * 1.05) ** 2)
            &
            (dist_sq <= (radius * 1.7) ** 2)
        )

        if (
            not np.any(center_region)
            or not np.any(ring_region)
            or not np.any(robust_ring_region)
        ):
            return None

        center_intensity = float(
            np.mean(patch[center_region])
        )

        # 기존 값은 그대로 유지
        ring_intensity = float(
            np.percentile(
                patch[ring_region],
                30,
            )
        )

        center_ring_contrast = (
            center_intensity
            - ring_intensity
        )

        # 검은 혈관 몇 개 때문에 값이 낮아지지 않도록
        # 좀 더 밝은 쪽(60 percentile)을 주변 기준으로 사용
        robust_ring_intensity = float(
            np.percentile(
                patch[robust_ring_region],
                60,
            )
        )

        robust_contrast = (
            center_intensity
            - robust_ring_intensity
        )

        angle = (
            np.arctan2(
                yy - cy,
                xx - cx,
            )
            + 2.0 * np.pi
        ) % (2.0 * np.pi)

        sector_contrasts = []

        num_sectors = 8

        for k in range(num_sectors):

            a0 = (
                2.0 * np.pi
                * k
                / num_sectors
            )

            a1 = (
                2.0 * np.pi
                * (k + 1)
                / num_sectors
            )

            sector_region = (
                robust_ring_region
                &
                (angle >= a0)
                &
                (angle < a1)
            )

            values = patch[sector_region]

            if values.size < 3:
                continue

            # 해당 방향에서 혈관 몇 pixel의 영향을 줄임
            sector_intensity = float(
                np.percentile(
                    values,
                    60,
                )
            )

            sector_contrasts.append(
                center_intensity
                - sector_intensity
            )

        if len(sector_contrasts) < 6:
            return None

        sector_contrasts = np.asarray(
            sector_contrasts,
            dtype=np.float32,
        )

        directional_support = float(
            np.mean(
                sector_contrasts > 0.0
            )
        )

        return (
            center_intensity,
            ring_intensity,
            center_ring_contrast,
            robust_contrast,
            absolute_center_intensity,
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