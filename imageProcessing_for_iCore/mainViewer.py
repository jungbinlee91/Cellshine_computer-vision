"""
CS_IMAGING — 술잔세포(Goblet Cell) 영상 분석 뷰어.

화면 구성 (2x2 패널):

  ┌─────────────────┬─────────────────┐
  │     RAW         │     LIVE        │
  │  비트 시프트만   │  ImageJ Auto    │
  │  (가공 없음)     │  B&C            │
  ├─────────────────┼─────────────────┤
  │  PROCESSING     │  DETECTING      │
  │  Blur+Stretch   │  Processing/Live│
  │  +CLAHE         │  배경 + 마커     │
  └─────────────────┴─────────────────┘

각 패널 알고리즘 상세:
  - RAW         : 원본 픽셀을 비트 시프트만 적용 (16→8 bit). 콘트라스트 조작 없음.
  - LIVE        : liveOptimizer.LiveOptimizer  (ImageJ Auto B&C 히스토그램 컷오프)
  - PROCESSING  : imageProc.ImageProcessor.apply_enhancement
                  (Gaussian Blur → 퍼센타일 선형 스트레칭 → CLAHE)
  - DETECTING   : 위의 PROCESSING 결과(또는 BG ON 시 LIVE) 위에
                  detectingGC.GobletDetector 검출 결과(컨투어/ROI 박스/카운트) 오버레이

조작:
  OPEN          - 이미지(.tif/.png/.jpg/.bmp) 열기 + 4개 패널 자동 갱신
  LIVE VIEW     - LIVE 패널만 재계산
  PROCESSING    - PROCESSING + DETECTING 재계산
  DETECTING     - DETECTING 만 재계산 (ROI/검출 파라미터 변경 시)
  BG ON         - DETECTING 배경: Processing(기본) ↔ LIVE
  MARK ON       - DETECTING 마커(ROI 박스 + 컨투어 + Count + ROI 크기) 표시
  SAVE          - 각 패널 현재 표시 이미지 저장
"""

import sys
import time

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from detectingGC import (
    FONT_SCALE,
    FONT_THICKNESS,
    MARKER_THICKNESS,
    ROI_THICKNESS,
    GobletDetector,
)
from detectingGC_auto import GobletDetector_Auto

from imageProc import ImageProcessor
from imageProc_auto import ImageProcessor_Auto
from liveOptimizer import LiveOptimizer, detect_valid_bits


# ============================================================
# CS_IMAGING Dark Theme Stylesheet
# ============================================================
DARK_QSS = """
QMainWindow, QWidget { background-color: #0d0d12; color: #d0d0d0; font-family: 'Segoe UI', sans-serif; font-size: 11px; }
QScrollArea { background-color: #0d0d12; border: none; }
QGroupBox {
    background-color: #14141a;
    border: 1px solid #2a2a35;
    border-radius: 4px;
    margin-top: 14px;
    padding: 12px 8px 8px 8px;
    font-weight: bold;
    color: #c8c8c8;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #d8d8d8; }
QPushButton {
    background-color: #1c1c25;
    color: #e0e0e0;
    border: 1px solid #2a2a35;
    border-radius: 3px;
    padding: 6px 10px;
}
QPushButton:hover { background-color: #25252f; border-color: #3a3a48; }
QPushButton:pressed { background-color: #15151c; }
QPushButton#topBtn { background-color: #1a1a22; padding: 8px; font-weight: bold; }
QPushButton#toggleBtn { background-color: #2a8ad0; color: white; font-weight: bold; padding: 4px 12px; border: none; border-radius: 3px; }
QPushButton#toggleBtn:checked { background-color: #444; color: #aaa; }
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #0a0a10;
    color: #e8e8e8;
    border: 1px solid #2a2a35;
    border-radius: 2px;
    padding: 2px 4px;
    min-height: 18px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #1c1c25; border: 1px solid #2a2a35; width: 14px;
}
QLabel { color: #c0c0c0; }
QLabel#headerLabel { color: #e84545; font-weight: bold; font-size: 13px; padding: 4px 8px; }
QLabel#fpsLabel { color: #e84545; font-weight: bold; padding: 4px 8px; }
QLabel#statusLabel { color: #e84545; font-weight: bold; padding: 4px 8px; }
QTextEdit, QPlainTextEdit {
    background-color: #0a0a10; color: #c0c0c0;
    border: 1px solid #2a2a35; border-radius: 3px;
}
QCheckBox { color: #c0c0c0; }
QSplitter::handle { background-color: #1a1a22; }
QFrame#panelFrame { background-color: #0a0a10; border: 1px solid #2a2a35; border-radius: 3px; }
"""


# ============================================================
# Image Panel (RAW / LIVE / PROCESSING / DETECTING)
# ============================================================
class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #000; border: none;")
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._pixmap = None

    def set_pixmap(self, pixmap):
        self._pixmap = pixmap
        self.update_view()

    def update_view(self):
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            super().setPixmap(scaled)

    def resizeEvent(self, e):
        self.update_view()


class ViewerPanel(QFrame):
    """CS_IMAGING-style panel: header (title + optional fps + togglable buttons),
    image area, footer (status + optional SAVE)."""
    def __init__(self, title, show_save=True, show_fps=False):
        super().__init__()
        self.setObjectName("panelFrame")
        self.setFrameShape(QFrame.Shape.NoFrame)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        self._header_layout = QHBoxLayout()
        self._header_layout.setContentsMargins(0, 0, 0, 0)
        self.header = QLabel(title)
        self.header.setObjectName("headerLabel")
        self._header_layout.addWidget(self.header)
        self._header_layout.addStretch()

        if show_fps:
            self.fps_label = QLabel("0 fps")
            self.fps_label.setObjectName("fpsLabel")
            self._header_layout.addWidget(self.fps_label)
        else:
            self.fps_label = None

        outer.addLayout(self._header_layout)

        # Image area
        self.image = ImageLabel()
        outer.addWidget(self.image, 1)

        # Footer
        footer_h = QHBoxLayout()
        footer_h.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel("")
        self.status.setObjectName("statusLabel")
        footer_h.addWidget(self.status)
        footer_h.addStretch()

        if show_save:
            self.save_btn = QPushButton("SAVE")
            self.save_btn.setFixedWidth(70)
            footer_h.addWidget(self.save_btn)
        else:
            self.save_btn = None

        outer.addLayout(footer_h)

        self.toggles = {}

    def add_toggle(self, key, text, checked=True):
        """Append a checkable toggle button to the header; returns the button."""
        btn = QPushButton(text)
        btn.setObjectName("toggleBtn")
        btn.setCheckable(True)
        btn.setChecked(checked)
        self._header_layout.addWidget(btn)
        self._header_layout.addSpacing(4)
        self.toggles[key] = btn
        return btn

    def set_pixmap(self, pixmap):
        self.image.set_pixmap(pixmap)

    def set_status(self, text):
        self.status.setText(text)

    def set_fps(self, fps):
        if self.fps_label is not None:
            self.fps_label.setText(f"{fps} fps")


# ============================================================
# Main Window
# ============================================================
class GobletAnalyzerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CS_IMAGING")
        self.setGeometry(50, 50, 1700, 1000)
        self.setStyleSheet(DARK_QSS)

        self.detector = GobletDetector()
        self.detector_auto = GobletDetector_Auto()
        self.live_opt = LiveOptimizer(step=10)

        self.raw_image = None           # 원본 (uint8 or uint16, 2D)
        self.raw_display = None         # RAW 패널용 8-bit (비트 시프트만)
        self.valid_bits = 8             # 로드된 이미지의 유효 비트 수
        self.res_live = None            # LIVE 패널 이미지 (uint8)
        self.res_normal = None          # PROCESSING 패널 이미지 (uint8)
        self.contours_n = []            # 검출된 GC 컨투어 리스트
        self.count_n = 0                # GC 개수
        self.elapsed_proc_ms = 0.0      # 마지막 PROCESSING 소요시간
        self.elapsed_det_ms = 0.0       # 마지막 DETECTING 소요시간
        # QImage 가 numpy 버퍼 raw 포인터를 참조하는 동안 GC 방지용 보관
        self._rgb_keepalive = {}
        self._panels_equalized = False

        self.init_ui()

    # --------------------------------------------------------
    # UI Construction
    # --------------------------------------------------------
    def init_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        main_layout = QHBoxLayout(cw)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # ----- Left sidebar -----
        scroll = QScrollArea()
        scroll.setFixedWidth(360)
        scroll.setWidgetResizable(True)
        side = QWidget()
        v = QVBoxLayout(side)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        # Top buttons
        self.btn_open = QPushButton("OPEN")
        self.btn_live = QPushButton("LIVE VIEW")
        self.btn_proc = QPushButton("PROCESSING")
        self.btn_det = QPushButton("DETECTING")
        for b in (self.btn_open, self.btn_live, self.btn_proc, self.btn_det):
            b.setObjectName("topBtn")
            v.addWidget(b)
        self.btn_open.clicked.connect(self.open_image)
        self.btn_live.clicked.connect(self.run_live)
        self.btn_proc.clicked.connect(self.run_processing)
        self.btn_det.clicked.connect(self.run_detecting)

        # File info box
        self.file_info = QTextEdit("No file loaded.")
        self.file_info.setReadOnly(True)
        self.file_info.setFixedHeight(90)
        v.addWidget(self.file_info)

        # ---------- LIVE VIEW group ----------
        g_live = QGroupBox("LIVE VIEW")
        gl = QVBoxLayout(g_live)
        gl.addWidget(QLabel("Auto Brightness/Contrast\n(no parameters)"))
        self.btn_live_apply = QPushButton("APPLY")
        self.btn_live_apply.clicked.connect(self.run_live)
        gl.addWidget(self.btn_live_apply)
        v.addWidget(g_live)

        # ---------- PROCESSING group (algorithm params kept intact) ----------
        g_proc = QGroupBox("PROCESSING")
        gp = QGridLayout(g_proc)
        gp.setHorizontalSpacing(6)
        gp.setVerticalSpacing(4)

        self.in_blur   = self._add_double(gp, 0, "Gaussian Blur",  5.0,  0.0, 50.0, 0.1)
        self.in_p_low  = self._add_double(gp, 1, "Low Stretch %",  1.0,  0.0, 100.0, 0.1)
        self.in_p_high = self._add_double(gp, 2, "High Stretch %", 99.9, 0.0, 100.0, 0.1)
        self.in_clahe  = self._add_double(gp, 3, "CLAHE Limit",    25.0, 0.0, 100.0, 0.1)

        h_pr = QHBoxLayout()
        self.btn_proc_reset = QPushButton("Reset")
        self.btn_proc_auto = QPushButton("AUTO")
        self.btn_proc_apply = QPushButton("APPLY")
        self.btn_proc_reset.clicked.connect(self.reset_processing_defaults)
        # AUTO 알고리즘 구현 전까지 APPLY와 동일한 동작을 수행합니다.
        self.btn_proc_auto.clicked.connect(self.run_processing_auto)
        self.btn_proc_apply.clicked.connect(self.run_processing)
        h_pr.addWidget(self.btn_proc_reset)
        h_pr.addWidget(self.btn_proc_auto)
        h_pr.addWidget(self.btn_proc_apply)
        gp.addLayout(h_pr, 4, 0, 1, 2)
        v.addWidget(g_proc)

        # ---------- DETECTING group ----------
        g_det = QGroupBox("DETECTING")
        gd = QGridLayout(g_det)
        gd.setHorizontalSpacing(6)
        gd.setVerticalSpacing(4)

        self.chk_marker = QCheckBox("Show GC Markers")
        self.chk_marker.setChecked(True)
        self.chk_marker.stateChanged.connect(self.refresh_display)
        gd.addWidget(self.chk_marker, 0, 0, 1, 2)

        self.in_contrast = self._add_double(gd, 1, "Contrast Th",    5.0,  0.0, 100.0, 0.1)
        self.in_min_s    = self._add_double(gd, 2, "Min Size (%)",   0.3,  0.0, 100.0, 0.1)
        self.in_max_s    = self._add_double(gd, 3, "Max Size (%)",   1.0,  0.0, 100.0, 0.1)
        self.in_block    = self._add_double(gd, 4, "Block Size (%)", 1.0,  0.1, 100.0, 0.1)
        self.in_adaptive = self._add_double(gd, 5, "Adaptive C",     0.5, -50.0, 50.0, 0.1)
        self.in_sens     = self._add_double(gd, 6, "Sensitivity",    45.0, 0.0, 100.0, 0.5)
        self.in_circ     = self._add_double(gd, 7, "Circularity",    20.0, 0.0, 100.0, 0.5)

        h_det = QHBoxLayout()
        self.btn_det_reset = QPushButton("Reset")
        self.btn_det_auto = QPushButton("AUTO")
        self.btn_det_apply = QPushButton("APPLY")
        self.btn_det_reset.clicked.connect(self.reset_detecting_defaults)
        # AUTO 알고리즘 구현 전까지 APPLY와 동일한 동작을 수행합니다.
        self.btn_det_auto.clicked.connect(self.run_detecting_auto)
        self.btn_det_apply.clicked.connect(self.run_detecting)
        h_det.addWidget(self.btn_det_reset)
        h_det.addWidget(self.btn_det_auto)
        h_det.addWidget(self.btn_det_apply)
        gd.addLayout(h_det, 8, 0, 1, 2)
        v.addWidget(g_det)

        # ---------- ROI group ----------
        g_roi = QGroupBox("ROI")
        gr = QGridLayout(g_roi)
        gr.setHorizontalSpacing(6)
        gr.setVerticalSpacing(4)
        self.in_fov_w  = self._add_double(gr, 0, "FOV width (mm)",   2.600, 0.001, 100.0, 0.001, 3)
        self.in_fov_h  = self._add_double(gr, 1, "FOV height (mm)",  1.699, 0.001, 100.0, 0.001, 3)
        self.in_roi_w  = self._add_double(gr, 2, "ROI width (mm)",   0.500, 0.001, 100.0, 0.001, 3)
        self.in_roi_h  = self._add_double(gr, 3, "ROI height (mm)",  0.500, 0.001, 100.0, 0.001, 3)
        self.in_roi_cx = self._add_double(gr, 4, "ROI center X (0..1)", 0.500, 0.0, 1.0, 0.01, 3)
        self.in_roi_cy = self._add_double(gr, 5, "ROI center Y (0..1)", 0.500, 0.0, 1.0, 0.01, 3)
        v.addWidget(g_roi)

        v.addStretch()
        scroll.setWidget(side)
        main_layout.addWidget(scroll)

        # ----- Right side: 2x2 grid of panels -----
        self.right_split_v = QSplitter(Qt.Orientation.Vertical)

        self.top_split_h = QSplitter(Qt.Orientation.Horizontal)
        self.panel_raw  = ViewerPanel("RAW",  show_save=False, show_fps=False)
        self.panel_live = ViewerPanel("LIVE", show_save=True,  show_fps=True)
        self.panel_live.save_btn.clicked.connect(lambda: self._save_image(self.res_live, "live"))
        self.top_split_h.addWidget(self.panel_raw)
        self.top_split_h.addWidget(self.panel_live)
        self.top_split_h.setStretchFactor(0, 1)
        self.top_split_h.setStretchFactor(1, 1)

        self.bot_split_h = QSplitter(Qt.Orientation.Horizontal)
        self.panel_proc = ViewerPanel("PROCESSING", show_save=True)
        self.panel_det  = ViewerPanel("DETECTING",  show_save=True)
        # DETECTING toggles: BG ON swaps background (Processing ↔ Live), MARK ON overlays markers
        self.btn_bg   = self.panel_det.add_toggle("bg",   "BG ON",   checked=False)
        self.btn_mark = self.panel_det.add_toggle("mark", "MARK ON", checked=True)
        self.btn_bg.toggled.connect(self.refresh_display)
        self.btn_mark.toggled.connect(self.refresh_display)
        self.panel_proc.save_btn.clicked.connect(lambda: self._save_image(self.res_normal, "processing"))
        self.panel_det.save_btn.clicked.connect(lambda: self._save_image(self._detecting_composite(), "detecting"))
        self.bot_split_h.addWidget(self.panel_proc)
        self.bot_split_h.addWidget(self.panel_det)
        self.bot_split_h.setStretchFactor(0, 1)
        self.bot_split_h.setStretchFactor(1, 1)

        self.right_split_v.addWidget(self.top_split_h)
        self.right_split_v.addWidget(self.bot_split_h)
        self.right_split_v.setStretchFactor(0, 1)
        self.right_split_v.setStretchFactor(1, 1)

        main_layout.addWidget(self.right_split_v, 1)

    # --------------------------------------------------------
    # Helper: 라벨 + double spinbox 한 줄 추가
    # --------------------------------------------------------
    def _add_double(self, grid, row, label, default, mn, mx, step, decimals=2):
        grid.addWidget(QLabel(label), row, 0)
        sb = QDoubleSpinBox()
        sb.setRange(mn, mx)
        sb.setSingleStep(step)
        sb.setDecimals(decimals)
        sb.setValue(default)
        sb.setFixedWidth(95)
        grid.addWidget(sb, row, 1, alignment=Qt.AlignmentFlag.AlignRight)
        return sb

    # --------------------------------------------------------
    # Reset helpers
    # --------------------------------------------------------
    def reset_processing_defaults(self):
        self.in_blur.setValue(5.0)
        self.in_p_low.setValue(1.0)
        self.in_p_high.setValue(99.9)
        self.in_clahe.setValue(25.0)
        self.run_processing()

    def reset_detecting_defaults(self):
        self.in_contrast.setValue(5.0)
        self.in_min_s.setValue(0.3)
        self.in_max_s.setValue(1.0)
        self.in_block.setValue(1.0)
        self.in_adaptive.setValue(0.5)
        self.in_sens.setValue(45.0)
        self.in_circ.setValue(20.0)
        self.run_detecting()

    # --------------------------------------------------------
    # File I/O
    # --------------------------------------------------------
    def open_image(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.tif *.tiff *.bmp)"
        )
        if not f:
            return
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.raw_image = img
        self.valid_bits = detect_valid_bits(img)
        self.raw_display = self._build_raw_display(img, self.valid_bits)
        h, w = self.raw_image.shape[:2]
        self.file_info.setPlainText(
            f"Loaded:\n{f}\n\nSize: {w} x {h}\nDType: {self.raw_image.dtype}\n"
            f"Valid bits: {self.valid_bits}"
        )
        # 4개 패널 자동 갱신
        self.run_live()
        self.run_processing_auto()
        self.run_detecting_auto()

    @staticmethod
    def _build_raw_display(img, valid_bits):
        """RAW 패널용 8-bit view. 콘트라스트 스트레칭 없이 비트 시프트만."""
        if img.dtype == np.uint8:
            return img
        if img.dtype == np.uint16:
            shift = max(0, valid_bits - 8)
            return (img >> shift).astype(np.uint8)
        return img.astype(np.uint8)

    def _save_image(self, img, tag):
        if img is None:
            return
        f, _ = QFileDialog.getSaveFileName(
            self, f"Save {tag}", f"{tag}.png", "Images (*.png *.tif *.jpg)"
        )
        if not f:
            return
        cv2.imwrite(f, img)

    # --------------------------------------------------------
    # Pipeline stages
    # --------------------------------------------------------
    def run_live(self):
        """LIVE 패널: ImageJ Auto B&C."""
        if self.raw_image is None:
            return
        t0 = time.time()
        self.res_live = self.live_opt.run(self.raw_image, valid_bits=self.valid_bits)
        dt = (time.time() - t0) * 1000.0
        fps = int(1000.0 / dt) if dt > 0 else 0
        self.panel_live.set_fps(fps)
        self.refresh_display()

    def run_processing(self):
        """PROCESSING 패널: Blur + 퍼센타일 스트레칭 + CLAHE. 이어서 DETECTING 재계산."""
        if self.raw_image is None:
            return
        t0 = time.time()
        self.res_normal = ImageProcessor.apply_enhancement(
            self.raw_image,
            float(self.in_p_low.value()), float(self.in_p_high.value()),
            float(self.in_blur.value()), float(self.in_clahe.value()),
        )
        self.elapsed_proc_ms = (time.time() - t0) * 1000.0
        # PROCESSING 결과가 바뀌었으므로 검출 재실행
        self.run_detecting_auto()

    def run_processing_auto(self):
            """PROCESSING 패널: Blur + 퍼센타일 스트레칭 + CLAHE. 이어서 DETECTING 재계산."""
            if self.raw_image is None:
                return
            t0 = time.time()
            self.res_normal = ImageProcessor_Auto.apply_adaptive_enhancement(
                self.raw_image,
                )
            self.elapsed_proc_ms = (time.time() - t0) * 1000.0
            # PROCESSING 결과가 바뀌었으므로 검출 재실행
            self.run_detecting_auto()

    def run_detecting(self):
        """DETECTING 패널: 술잔세포 검출 + 오버레이."""
        if self.res_normal is None:
            self.run_processing()
            return
        rx, ry, rw, rh = self._compute_roi(self.res_normal.shape)
        params = {
            "Contrast":    float(self.in_contrast.value()),
            "MinSize":     float(self.in_min_s.value()),
            "MaxSize":     float(self.in_max_s.value()),
            "BlockSize":   float(self.in_block.value()),
            "AdaptiveC":   float(self.in_adaptive.value()),
            "Sensitivity": float(self.in_sens.value()),
            "Circularity": float(self.in_circ.value()),
            "ROI_X": rx, "ROI_Y": ry, "ROI_W": rw, "ROI_H": rh,
        }
        t0 = time.time()
        self.contours_n, self.count_n, _ = self.detector.detect(self.res_normal, params)
        self.elapsed_det_ms = (time.time() - t0) * 1000.0
        self.refresh_display()

    def run_detecting_auto(self):
            """DETECTING 패널: 술잔세포 검출 + 오버레이."""
            if self.res_normal is None:
                self.run_processing_auto()
                return
            rx, ry, rw, rh = self._compute_roi(self.res_normal.shape)
            params = {
                "Contrast":    float(self.in_contrast.value()),
                "MinSize":     float(self.in_min_s.value()),
                "MaxSize":     float(self.in_max_s.value()),
                "BlockSize":   float(self.in_block.value()),
                "AdaptiveC":   float(self.in_adaptive.value()),
                "Sensitivity": float(self.in_sens.value()),
                "Circularity": float(self.in_circ.value()),
                "ROI_X": rx, "ROI_Y": ry, "ROI_W": rw, "ROI_H": rh,
            }
            t0 = time.time()
            self.contours_n, self.count_n, _ = self.detector_auto.detect_auto(self.res_normal, params)
            self.elapsed_det_ms = (time.time() - t0) * 1000.0
            self.refresh_display()

    # --------------------------------------------------------
    # ROI: (ROI_mm / FOV_mm) * 이미지 픽셀 → 픽셀 ROI 박스
    # --------------------------------------------------------
    def _compute_roi(self, shape):
        h_px, w_px = shape[:2]
        fov_w = max(float(self.in_fov_w.value()), 1e-6)
        fov_h = max(float(self.in_fov_h.value()), 1e-6)
        rw = int((float(self.in_roi_w.value()) / fov_w) * w_px)
        rh = int((float(self.in_roi_h.value()) / fov_h) * h_px)
        cx = float(self.in_roi_cx.value())
        cy = float(self.in_roi_cy.value())
        rx = int(cx * w_px - rw / 2)
        ry = int(cy * h_px - rh / 2)
        # 이미지 경계 클램프
        rx = max(0, min(rx, max(0, w_px - rw)))
        ry = max(0, min(ry, max(0, h_px - rh)))
        return rx, ry, rw, rh

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------
    def refresh_display(self):
        # RAW — bit-shifted only, no contrast stretching
        if self.raw_display is not None:
            self.panel_raw.set_pixmap(self._gray_to_pixmap(self.raw_display, "raw"))

        # LIVE — ImageJ Auto B&C (LiveOptimizer)
        if self.res_live is not None:
            self.panel_live.set_pixmap(self._gray_to_pixmap(self.res_live, "live"))

        # PROCESSING — enhanced image (Blur + percentile stretch + CLAHE)
        if self.res_normal is not None:
            self.panel_proc.set_pixmap(self._gray_to_pixmap(self.res_normal, "proc"))
            self.panel_proc.set_status(f"{self.elapsed_proc_ms/1000.0:.2f} sec")

        # DETECTING — layer order: Processing (bottom) → Live (if BG ON) → Mark (if MARK ON)
        disp = self._detecting_composite()
        if disp is not None:
            self.panel_det.set_pixmap(self._rgb_to_pixmap(disp, "det"))
            self.panel_det.set_status(f"{self.elapsed_det_ms/1000.0:.2f} sec | count: {self.count_n}")

    def _detecting_composite(self):
        """Compose the DETECTING panel image: chosen background + optional markers."""
        # Choose background: Live if BG ON and available, else Processing
        if self.btn_bg.isChecked() and self.res_live is not None:
            base = self.res_live
        else:
            base = self.res_normal
        if base is None:
            return None

        disp = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
        if self.chk_marker.isChecked() and self.btn_mark.isChecked():
            rx, ry, rw, rh = self._compute_roi(base.shape)
            NEON_GREEN = (0, 255, 0)  # fluorescent green (RGB)
            BLUE = (0, 0, 255)        # GC contour color

            # ROI rectangle
            cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), NEON_GREEN, ROI_THICKNESS)
            # Goblet cell contours (blue)
            cv2.drawContours(disp, self.contours_n, -1, BLUE, MARKER_THICKNESS)

            # Count label above the rectangle
            count_text = f"Count: {self.count_n}"
            (_, ch), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
            count_y = max(ch + 5, ry - 20)
            cv2.putText(disp, count_text, (rx, count_y),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, NEON_GREEN, FONT_THICKNESS)

            # ROI size label below the rectangle
            roi_w_mm = float(self.in_roi_w.value())
            roi_h_mm = float(self.in_roi_h.value())
            size_text = f"ROI: {roi_w_mm:.2f} x {roi_h_mm:.2f} mm"
            (_, sh), _ = cv2.getTextSize(size_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
            size_y = min(base.shape[0] - 10, ry + rh + sh + 20)
            cv2.putText(disp, size_text, (rx, size_y),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, NEON_GREEN, FONT_THICKNESS)
        return disp

    # --------------------------------------------------------
    # Pixmap 변환 (numpy 버퍼 수명 보장)
    # --------------------------------------------------------
    def _gray_to_pixmap(self, gray, key):
        """8-bit grayscale → QPixmap (Format_Grayscale8, cvtColor 불필요)."""
        g = np.ascontiguousarray(gray, dtype=np.uint8)
        self._rgb_keepalive[key] = g
        h, w = g.shape[:2]
        qimg = QImage(g.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)

    def _rgb_to_pixmap(self, rgb, key):
        """HxWx3 uint8 RGB → QPixmap."""
        r = np.ascontiguousarray(rgb, dtype=np.uint8)
        self._rgb_keepalive[key] = r
        h, w = r.shape[:2]
        qimg = QImage(r.data, w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    # --------------------------------------------------------
    # 첫 표시 시 4개 패널 크기 균등 분할
    # --------------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        if not self._panels_equalized:
            self._panels_equalized = True
            # 이벤트 루프 다음 틱에 호출 (스플리터 폭 확정 후)
            QTimer.singleShot(0, self._equalize_panels)

    def _equalize_panels(self):
        w_top = max(self.top_split_h.width() // 2, 1)
        self.top_split_h.setSizes([w_top, w_top])
        w_bot = max(self.bot_split_h.width() // 2, 1)
        self.bot_split_h.setSizes([w_bot, w_bot])
        h_right = max(self.right_split_v.height() // 2, 1)
        self.right_split_v.setSizes([h_right, h_right])


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GobletAnalyzerUI()
    window.show()
    sys.exit(app.exec())
