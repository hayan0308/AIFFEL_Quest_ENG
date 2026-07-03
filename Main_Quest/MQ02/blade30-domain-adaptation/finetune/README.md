# blade30 도메인 적응 파인튜닝 파이프라인

`best.pt`(NordTank 도메인 학습)를 blade30(근접 고해상도) 도메인에 맞춰 성능을 끌어올리는 정공법 파이프라인.

## 무엇을, 왜

| 단계 | 스크립트 | 해결하는 문제 |
|------|----------|---------------|
| ① 타일링 | `step1_tile_blade30.py` | **스케일 불일치** — 고해상도 원본을 학습 해상도(640) 타일로 분할. 109장→수백 타일로 확장(실제 픽셀). 빈 타일 일부를 target 배경으로 보존 |
| ②③ 구성 | `step2_build_dataset.py` | **망각 방지 + 오탐 억제** — blade30 타일 + NordTank 라벨 + NordTank 정상이미지(배경) 혼합 |
| ② 학습 | `step3_finetune.py` | **도메인 적응** — best.pt에서 전이학습(백본 동결·낮은 LR·표적 증강·copy_paste로 dirt 보강 ④⑤) |
| ④ 평가 | `step4_evaluate.py` | **정직한 측정** — blade30 val에서 파인튜닝 전/후 mAP 비교 |

핵심 설계 근거:
- **누수 방지**: blade 그룹(예: `Blade_5`) 단위로 train/val 분할 → 같은 원본의 타일이 양쪽에 섞이지 않음
- **배경(negative)**: 라벨 없는 정상 이미지를 넣어 "깨끗한 표면=결함 아님"을 학습 → 앞서 본 오탐 감소
- **클래스 불균형(dirt 62 vs damage 6295)**: `copy_paste` 증강으로 희귀 dirt 보강

## 실행 순서

```bash
# (가상환경 활성화)
source ../.venv/bin/activate
cd finetune

python step1_tile_blade30.py     # CPU 빠름 — 타일 생성
python step2_build_dataset.py    # CPU 빠름 — finetune.yaml 생성
python step3_finetune.py         # ★ GPU 권장 (CPU는 매우 느림)
python step4_evaluate.py         # 전/후 mAP 비교
```

## 설정 조정

모든 경로·하이퍼파라미터는 `config.py` 한 곳에 있음. 자주 만질 것:
- `TILE` (640) — 학습 imgsz와 반드시 일치. 크게 하면 문맥↑·속도↓
- `NEG_RATIO` (1.0) — 배경 타일 비율. 오탐이 많으면 ↑, recall이 낮으면 ↓
- `NORDTANK_LABELED_MAX` / `NORDTANK_BG_MAX` — 혼합 비율. blade30에 더 치중하려면 NordTank 수를 줄임
- `step3` 의 `freeze`, `lr0`, `epochs`, `batch` — GPU 메모리/데이터 양에 맞게

## 현재 검증된 산출물 (이 맥에서 step1~2 실행 결과)
- blade30 타일: train 440 (양성 220 + 배경 220), val 124
- 최종 train 4,935 (blade30 440 + NordTank 라벨 2,995 + 배경 1,500), val 124
- step3(학습)만 GPU 머신에서 실행하면 됨
