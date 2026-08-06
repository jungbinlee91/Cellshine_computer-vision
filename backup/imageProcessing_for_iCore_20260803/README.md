# CS_IMAGING — 술잔세포(Goblet Cell) 영상 분석 뷰어

전안부(결막) 영상에서 술잔세포를 자동 검출·시각화하는 PyQt6 기반 데스크탑 도구.

---

## 1. 실행 환경

### 1-1. 개발/검증 환경 (납품 시점)

| 항목 | 버전 |
|---|---|
| OS            | Windows 11 (x64) |
| Python        | **3.14.4** (CPython, 64-bit) |
| PyQt6         | 6.11.0 (Qt 런타임 6.11.0 포함) |
| opencv-python | 4.13.0 |
| NumPy         | 2.4.4 |

> Python 3.10 ~ 3.14 호환. macOS / Linux 에서도 동일 패키지로 동작 가능.

### 1-2. 사용 오픈소스 라이선스

| 패키지 | 라이선스 | 용도 |
|---|---|---|
| [Python](https://www.python.org/)            | PSF License             | 런타임 |
| [PyQt6](https://riverbankcomputing.com/software/pyqt/) | GPL v3 / 상용 라이선스 (Riverbank) | GUI 프레임워크 |
| [Qt 6](https://www.qt.io/)                   | LGPL v3 / 상용          | PyQt6 가 의존하는 C++ 라이브러리 |
| [OpenCV (opencv-python)](https://opencv.org/) | Apache 2.0              | 이미지 처리·검출 알고리즘 |
| [NumPy](https://numpy.org/)                  | BSD 3-Clause            | 배열 연산 |

> **상업적 배포 시 주의**: PyQt6 는 GPL v3 라이선스로 제공됩니다. 본 소프트웨어를 상업 라이선스 조건으로 재배포하려면 Riverbank Computing 의 PyQt 상용 라이선스 또는 LGPL 기반 대안(PySide6) 사용을 검토하세요.

### 1-3. 다른 컴퓨터에 설치하기 (Windows)

#### Step 1 — Python 3.14 설치
1. [python.org/downloads](https://www.python.org/downloads/) 에서 **Python 3.14.x Windows installer (64-bit)** 다운로드
2. 설치 시 **"Add Python to PATH"** 체크
3. 설치 후 새 PowerShell 창에서 확인:
   ```powershell
   python --version
   # Python 3.14.4
   ```

#### Step 2 — 프로젝트 폴더 복사
이 폴더(`0.8.4`) 통째로 원하는 위치에 복사. 예: `C:\CS_IMAGING\`

#### Step 3 — 가상환경 생성 (권장)
시스템 Python 을 더럽히지 않도록 프로젝트 폴더 안에 격리된 환경 생성:

```powershell
cd C:\CS_IMAGING\0.8.4
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> PowerShell 실행 정책 오류가 나면 한 번만:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

활성화 성공 시 프롬프트 앞에 `(.venv)` 표시.

#### Step 4 — 의존성 일괄 설치
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

설치 확인:
```powershell
python -c "import cv2, numpy, PyQt6.QtCore as q; print('cv2', cv2.__version__, '/ numpy', numpy.__version__, '/ PyQt6', q.PYQT_VERSION_STR)"
```

#### Step 5 — 실행
```powershell
python mainViewer.py
```

### 1-4. 두 번째 실행부터 (간단 모드)

PowerShell 새 창에서:
```powershell
cd C:\CS_IMAGING\0.8.4
.\.venv\Scripts\Activate.ps1
python mainViewer.py
```

### 1-5. 바탕화면 바로가기 만들기 (선택)

새 `.bat` 파일을 만들어 바탕화면에 두면 더블클릭으로 실행:

```bat
@echo off
cd /d C:\CS_IMAGING\0.8.4
call .venv\Scripts\activate.bat
python mainViewer.py
```

### 1-6. 입력 포맷
- `.tif / .tiff / .png / .jpg / .bmp` 지원
- uint8 또는 uint16 grayscale
- 16-bit TIFF 가 실제로는 12-bit 카메라 데이터(예: STC-MBS2041)인 경우 자동 감지

---

## 2. 화면 구성

좌측 사이드바: `OPEN / LIVE VIEW / PROCESSING / DETECTING` 버튼 + 파일 정보 + 파라미터 그룹.

우측 2×2 패널:

| | 좌 | 우 |
|---|---|---|
| 상 | **RAW** — 원본 (가공 없음) | **LIVE** — 자동 명암 |
| 하 | **PROCESSING** — 강조 처리 | **DETECTING** — 검출 결과 |

---

## 3. 각 패널의 이미지 처리

### 3-1. RAW 패널
원본 픽셀을 **콘트라스트 조작 없이** 그대로 표시.

- `uint8` : 그대로 표시
- `uint16`: `raw >> (valid_bits − 8)` 비트 시프트만 (예: 12-bit 데이터는 `>> 4`)
- `valid_bits` 는 데이터 실측 최대값 `ceil(log2(max+1))` 으로 자동 추정

### 3-2. LIVE 패널 — ImageJ Auto Brightness/Contrast
ImageJ `ContrastAdjuster.java` 의 Auto B&C 와 동일한 히스토그램 기반 자동 컷오프.

1. 전체 히스토그램 계산 (uint8 → 256 bins, uint16 → `1<<valid_bits` bins)
2. **Saturation guard**: 단일 bin 픽셀 수가 전체의 10% 초과 → 포화/배경으로 간주, 컷오프 계산에서 제외
3. `threshold = pixelCount / 5000`: 이보다 많은 픽셀이 있는 bin 중 양 끝의 `low`, `high` 결정
4. `LUT[v] = clip((v − low) × 255 / (high − low), 0, 255)`
5. 출력 = `LUT[input]` (입력과 같은 shape, uint8)

파라미터 없음. **APPLY** 또는 좌상단 `LIVE VIEW` 버튼으로 재계산.

### 3-3. PROCESSING 패널 — 영상 강조
3단 파이프라인:

```
원본 ──► Gaussian Blur (5x5, σ = Gaussian Blur)
      ──► 퍼센타일 선형 스트레칭 (Low Stretch % ~ High Stretch %)
              · 10x10 서브샘플 히스토그램에서 두 백분위 값 추출
              · 그 구간을 [0, 255] 8-bit 로 선형 매핑
      ──► CLAHE (8x8 타일, clipLimit = CLAHE Limit)
              · 국소 콘트라스트 향상
```

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| Gaussian Blur | 5.00 | 가우시안 표준편차 σ. 0 → blur 안 함 |
| Low Stretch % | 1.00 | 어두운 쪽 컷오프 백분위 |
| High Stretch % | 99.90 | 밝은 쪽 컷오프 백분위 |
| CLAHE Limit | 25.00 | CLAHE clipLimit (작을수록 약함) |

### 3-4. DETECTING 패널 — 술잔세포 검출
PROCESSING 결과 위에서 **ROI 영역만** 검출. 단계별:

```
1) ROI 추출  (ROI mm 좌표 → 픽셀 좌표로 환산)
2) Median Blur (5x5)                            ─ 점잡음 제거
3) Adaptive Threshold (Gaussian, Block, AdaC)  ─ 지역적 이진화
4) Watershed 분할 (distance transform)          ─ 군집 객체 분리
5) 컨투어 후처리 필터
     · 면적 ∈ [Min Size%, Max Size%]² × ((W+H)/2)²
     · 원형도 (4π·A/P²) ≥ Circularity / 100
     · 컨투어 내부 평균 명도 − 주변 10px ring 평균 ≥ Contrast Th
```

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| Contrast Th | 5.00 | 컨투어 내·외부 명도 차이 최소값 |
| Min Size (%) | 0.30 | 최소 면적 (이미지 변 평균 대비 %)² |
| Max Size (%) | 1.00 | 최대 면적 (이미지 변 평균 대비 %)² |
| Block Size (%) | 1.00 | adaptive threshold block size (%) |
| Adaptive C | 0.50 | adaptive threshold C 상수 |
| Sensitivity | 45.00 | watershed distance transform 임계 비율 |
| Circularity | 20.00 | 최소 원형도 (%) |

#### DETECTING 패널 토글
| 토글 | 효과 |
|---|---|
| `BG ON` (기본 OFF) | 배경 이미지: OFF=PROCESSING, ON=LIVE |
| `MARK ON` (기본 ON) | ROI 박스 + 컨투어 + Count + ROI 크기 표시 |
| `Show GC Markers` (사이드바) | MARK ON 과 동시에 켜져야 마커 표시 (AND) |

오버레이 색상:
- **형광 녹색** : ROI 사각형, `Count: N`, `ROI: W × H mm`
- **파랑**      : GC 컨투어

---

## 4. ROI 계산식
사이드바 `ROI` 그룹의 mm 값과 이미지 해상도로 픽셀 ROI 박스를 계산:

```
ROI_w_px = (ROI width  / FOV width)  × image_width_px
ROI_h_px = (ROI height / FOV height) × image_height_px
center   = (ROI center X × image_w, ROI center Y × image_h)
좌상단   = center − (ROI_w_px/2, ROI_h_px/2)
```

이미지 경계를 넘으면 자동 클램프.

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| FOV width (mm)    | 2.600 | 카메라 시야 가로 폭 |
| FOV height (mm)   | 1.699 | 카메라 시야 세로 폭 |
| ROI width (mm)    | 0.500 | 분석 영역 가로 폭 |
| ROI height (mm)   | 0.500 | 분석 영역 세로 폭 |
| ROI center X (0..1) | 0.500 | ROI 중심 X (0=좌, 1=우) |
| ROI center Y (0..1) | 0.500 | ROI 중심 Y (0=상, 1=하) |

---

## 5. 조작 흐름

1. **OPEN** → 이미지 선택. 4개 패널 자동 계산.
2. PROCESSING 파라미터 조정 → **APPLY** (또는 좌상단 `PROCESSING` 버튼).
   - PROCESSING 갱신 시 DETECTING 도 자동 재실행됨.
3. DETECTING 파라미터/ROI 조정 → **APPLY** (또는 좌상단 `DETECTING` 버튼).
4. 각 패널의 **SAVE** 버튼으로 현재 표시 이미지 저장 (DETECTING 은 오버레이 포함).
5. **Reset** 으로 그룹별 기본값 복원.

---

## 6. 파일 구성

| 파일 | 역할 |
|---|---|
| `mainViewer.py`     | UI / 이벤트 / 파이프라인 오케스트레이션 (실행 진입점) |
| `liveOptimizer.py`  | LIVE 패널 — ImageJ Auto B&C 알고리즘 |
| `imageProc.py`      | PROCESSING 패널 — Blur+Stretch+CLAHE 알고리즘 |
| `detectingGC.py`    | DETECTING 패널 — 술잔세포 검출 알고리즘 + 오버레이 두께 상수 |
| `requirements.txt`  | 의존 패키지 버전 고정 (다른 PC 동일 환경 재현용) |
| `README.md`         | 이 문서 |

---

## 7. 문제 해결 (FAQ)

| 증상 | 해결 |
|---|---|
| `python` 명령을 찾을 수 없음 | Python 설치 시 "Add Python to PATH" 체크 누락. 재설치 또는 환경변수 PATH 에 `C:\Users\<user>\AppData\Local\Programs\Python\Python314\` 추가 |
| `Activate.ps1` 실행 거부됨 | PowerShell 관리자 권한으로 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 1회 실행 |
| `pip install` 시 ssl/네트워크 오류 | 사내 프록시 환경이면 `pip install --proxy http://<proxy>:<port> -r requirements.txt` |
| `ImportError: DLL load failed` (opencv) | Visual C++ Redistributable 미설치. [Microsoft 페이지](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) 에서 x64 버전 설치 |
| GUI 가 늦게 뜸 (5~15초) | VS Code 디버거 사용 시 정상. 일반 실행은 2~3초 |
| 16-bit TIFF 가 RAW 패널에서 너무 어둡게 보임 | 정상. RAW 는 비트 시프트만 적용. 자동 대비를 보려면 LIVE 패널 확인 |
