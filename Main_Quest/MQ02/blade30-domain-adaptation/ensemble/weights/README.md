# weights/

모델 가중치(.pt)는 용량이 커서 저장소에 포함하지 않습니다(.gitignore).
`ensemble_eval.py` / `run_ensemble.py` 를 돌리려면 아래 5개 파일을 이 폴더에 넣으세요:

- `yolov8s_norm1500.pt`  (단일 챔피언)
- `yolo11s_norm750.pt`
- `yolov10s_norm750.pt`
- `yolo12s_norm750.pt`
- `rtdetr-l.pt`

(팀 공유 드라이브 / 원본 `ensemble_bundle/weights/` 참조)
