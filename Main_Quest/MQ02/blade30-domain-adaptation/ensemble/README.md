# MQ02 앙상블 묶음 (WBF Ensemble) — 사용 안내

우리가 학습한 5개 모델(YOLO 4종 + rtdetr)을 **WBF(Weighted Boxes Fusion)** 로 합친 앙상블입니다.
학습은 안 하고 **추론/평가만** 합니다. 이 폴더 하나만 있으면 어디서든 돌아갑니다.

우리 test에서 damage recall이 단일 최고 0.856 -> 앙상블 0.944로 올랐습니다. 놓침을 줄이는 게 목적입니다.

## 0. 설치 (한 번만)
```
pip install ultralytics ensemble-boxes opencv-python
```

## 폴더 구성
```
ensemble_bundle/
├── run_ensemble.py     # (A) 라벨 없는 이미지 -> 박스 그림 + 예측 라벨
├── ensemble_eval.py    # (B) 라벨 있는 데이터 -> 점수(mAP/recall) 측정   ★이걸 쓰세요
├── weights/            # 모델 5개 (yolov8s/11s/v10s/12s + rtdetr)
├── input/              # 여기에 데이터를 넣거나, 스크립트의 DATA_DIR 을 바꾸세요
└── output/             # 결과가 여기 저장됨
```

## (B) 점수 측정 — 라벨 있는 데이터가 있을 때 (팀원용)

1. 데이터를 아래 구조로 준비 (YOLO 포맷):
   ```
   input/
     images/  사진들 (.png/.jpg)
     labels/  같은 이름 .txt  ("class cx cy w h" 정규화)
   ```
   (또는 `ensemble_eval.py` 맨 위 `DATA_DIR` 을 데이터 폴더로 바꿔도 됩니다.)

2. **★ 클래스 번호를 우리와 맞춰주세요: `0 = dirt(오염)`, `1 = damage(손상)`.**
   데이터가 손상을 crack/erosion/breakage 등으로 세분화했다면, 라벨 txt에서
   **손상 계열은 전부 `1`, 오염 계열은 `0`** 으로 바꾼(remap) 뒤 넣어야 비교가 맞습니다.

3. 실행:
   ```
   python ensemble_eval.py
   ```

4. 결과가 `output/` 에 저장됩니다:
   - **`output/results.txt`** — 단일 4종 vs 앙상블 점수표 (사람이 읽는 표)
   - **`output/summary.json`** — 같은 수치의 기계 판독용 (이미지 수, GT 개수 포함)

   **-> 이 두 파일(results.txt + summary.json)을 김민욱에게 보내면 됩니다.**
   (`ensemble_eval.py` 에서 `SAVE_VIS = True` 로 바꿔 돌리면 `output/vis/` 에 박스 그림도
    생깁니다. 대표 몇 장 같이 보내주면 눈으로도 볼 수 있어 좋습니다.)

## (A) 그림만 뽑기 — 라벨 없을 때
`run_ensemble.py` 의 `INPUT_DIR` 에 이미지 폴더를 주고 `python run_ensemble.py`.
`output/vis`(박스 그림) + `output/labels`(예측 라벨)가 나옵니다.

## 참고
- rtdetr(252MB)이 커서 전송이 부담되면, 스크립트 맨 위 `USE_RTDETR = False` 로 두고
  rtdetr-l.pt 를 빼도 됩니다. YOLO 4종만으로도 recall 0.891 나옵니다(0.944보다 약간 낮음).
- 지표는 conf>=0.05 기준 자체 계산이라 ultralytics 공식 val 절대값과 다를 수 있습니다.
  단일·앙상블에 같은 잣대라 **"앙상블이 오르나"** 상대비교는 유효합니다.
