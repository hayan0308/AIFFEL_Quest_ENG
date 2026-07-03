# 풍력 블레이드 결함 탐지 — 도메인 적응 연구 (blade30)

NordTank(드론·저해상도) 도메인에서 학습한 결함 탐지 모델(YOLO 계열 + WBF 앙상블)을
**다른 도메인인 blade30(근접·고해상도)** 에 적용했을 때의 성능 붕괴를 분석하고,
그 원인(도메인 갭)을 정량적으로 규명한 뒤 개선 방법을 정리한 저장소입니다.

- **클래스**: `0 = dirt(오염)`, `1 = damage(손상)`
- **학습 도메인**: NordTank 586×371 (원거리 드론)
- **평가 도메인**: blade30 (근접 고해상도 5456×3632)

## 핵심 발견

| 데이터 (앙상블 ENS_ALL, imgsz 640) | mAP50 | damage recall |
|------------------------------------|:-----:|:-------------:|
| 학습 도메인 (test301) | 0.863 | 0.944 |
| blade30 원본 (타일 X) | 0.016 | 0.418 |
| blade30 타일 (960²) | 0.021 | 0.405 |

1. **도메인 갭이 성능 붕괴의 주원인** — blade30은 원본에서도 mAP50 0.016으로 붕괴. 모델 문제가 아니라 도메인 문제.
2. **앙상블 우위는 낯선 도메인에서도 견고** — 단일 챔피언 대비 damage recall 2.4배(0.176→0.418).
3. **타일화는 소폭 도움**(0.016→0.021)이나 도메인 갭을 메우진 못함. 640 > 960.
4. **dirt는 사실상 실패** — 학습 데이터에서 dirt가 희귀(1:15)했던 클래스 불균형 탓.

자세한 격리 분석: [`reports/blade30_isolation_report.md`](reports/blade30_isolation_report.md)

## 폴더 구조

```
├── finetune/              # ★ 도메인 적응 파인튜닝 파이프라인 (제안 해법)
│   ├── config.py              설정 한 곳 관리
│   ├── step1_tile_blade30.py  고해상도 → 640 타일 (전처리)
│   ├── step2_build_dataset.py 타일 + NordTank + 배경 혼합 데이터 구성
│   ├── step3_finetune.py      best.pt 전이학습 (GPU 권장)
│   ├── step4_evaluate.py      전/후 mAP 비교
│   └── README.md
├── ensemble/              # WBF 앙상블 추론/평가 (가중치는 별도, weights/README.md 참고)
│   ├── ensemble_eval.py       라벨 있는 데이터 → mAP/recall 측정
│   └── run_ensemble.py        라벨 없는 데이터 → 박스 그림/예측
├── sahi/                  # SAHI 슬라이스 추론 실험 (재학습 없는 개선 시도)
│   ├── run_sahi.py / run_sahi_tune.py / eval_sahi_map.py
├── preprocessing/
│   └── make_tiles.py          blade30 원본 → 960² 타일 생성
├── visualization/
│   └── make_slide_visuals.py  챔피언 vs 앙상블 비교 그림
├── configs/               # 데이터 yaml
├── reports/               # 격리 분석 리포트 + 앙상블 원자료
└── results/               # 학습곡선 / 앙상블 시각화 이미지
```

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 데이터·가중치

용량 문제로 저장소에는 **코드·문서·결과 이미지만** 포함됩니다.
데이터셋(blade30, NordTank)과 모델 가중치(.pt)는 `.gitignore` 처리되어 있으니,
팀 공유 드라이브에서 받아 각 스크립트의 경로(또는 `finetune/config.py`)에 맞춰 배치하세요.
앙상블 가중치는 [`ensemble/weights/README.md`](ensemble/weights/README.md) 참고.

## 재현 순서 (파인튜닝)

```bash
cd finetune
python step1_tile_blade30.py     # CPU 빠름
python step2_build_dataset.py    # CPU 빠름
python step3_finetune.py         # GPU 권장
python step4_evaluate.py         # 전/후 mAP 비교
```
