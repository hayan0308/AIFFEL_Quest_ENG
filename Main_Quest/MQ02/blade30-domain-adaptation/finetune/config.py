"""공통 설정 — 경로/하이퍼파라미터를 여기서 한 곳에서 관리한다."""
import os

# 프로젝트 루트 (이 파일 기준 상위 폴더)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 원본 데이터 ----
BLADE30_DIR   = os.path.join(ROOT, "blade30_yolo_defects")   # target 도메인 (고해상도, 라벨 109장)
NORDTANK_DIR  = os.path.join(ROOT, "NordTank586x371")        # 학습 도메인 (저해상도)
BASE_WEIGHTS  = os.path.join(ROOT, "best.pt")                # 파인튜닝 시작점

# ---- 산출물 ----
WORK          = os.path.join(ROOT, "finetune")
TILES_DIR     = os.path.join(WORK, "dataset", "blade30_tiles640")  # 타일링 결과 (640 타일, 이예령의 960² blade30_tiles와 구분)
DATA_YAML     = os.path.join(WORK, "dataset", "finetune.yaml")   # 최종 학습 yaml
RUNS_DIR      = os.path.join(ROOT, "predict_runs")               # 학습/평가 결과

# ---- 클래스 ----
NAMES = {0: "dirt", 1: "damage"}

# ---- 타일링 파라미터 (전처리 ①) ----
TILE       = 640     # 타일 한 변 크기 (학습 imgsz와 맞춘다)
OVERLAP    = 0.2     # 타일 간 겹침 비율
MIN_VIS    = 0.30    # 잘린 박스가 타일 안에 이 비율 이상 남아야 라벨로 인정
NEG_RATIO  = 1.0     # 양성 타일 대비 유지할 배경(빈) 타일 비율 (blade30 도메인 negative)

# ---- 데이터 구성 파라미터 (③ 배경 투입 / 혼합) ----
NORDTANK_LABELED_MAX = 3000   # 섞을 NordTank 라벨 이미지 최대 수 (망각 방지)
NORDTANK_BG_MAX      = 1500   # 배경으로 넣을 NordTank 정상(무라벨) 이미지 수 (오탐 억제)

# ---- 분할 ----
VAL_FRACTION = 0.2   # blade30을 (blade 그룹 단위로) train/val 분할하는 비율
SEED = 42
