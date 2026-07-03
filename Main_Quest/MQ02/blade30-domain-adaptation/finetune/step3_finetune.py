"""
파인튜닝 ② : best.pt 를 시작점으로 혼합 데이터셋에 전이학습한다.

소량 target 데이터(blade30)에 맞춘 설정:
- freeze: 백본 앞부분 동결 -> 과적합 완화, 저수준 특징 보존
- lr0 낮게 + cosine + warmup + EMA + early stopping(patience)
- 도메인 정합용 표적 증강(⑤): HSV/밝기(카메라·조명 차이), flip, mosaic/copy-paste(희귀 dirt 보강)

GPU 사용 예:
    python step3_finetune.py            # device=0 자동
    (CPU에서도 돌지만 매우 느림)
"""
import config as C
from ultralytics import YOLO
import torch


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    model = YOLO(C.BASE_WEIGHTS)   # 학습된 가중치에서 이어서 학습
    model.train(
        data=C.DATA_YAML,
        epochs=100,
        patience=20,            # 20 epoch 개선 없으면 조기 종료
        imgsz=C.TILE,           # 타일 크기와 일치
        batch=16,               # GPU 메모리에 맞게 조정
        device=device,
        project=C.RUNS_DIR,
        name="blade30_finetune",

        # --- 전이학습 안정화 ---
        freeze=10,              # 앞 10개 레이어(백본 일부) 동결
        lr0=0.001,              # 낮은 학습률 (기본 0.01의 1/10)
        lrf=0.01,               # 최종 LR = lr0 * lrf (cosine)
        cos_lr=True,
        warmup_epochs=3,
        optimizer="AdamW",
        weight_decay=0.0005,

        # --- 도메인 정합용 증강 ⑤ ---
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # 색/밝기 변화 (카메라·조명 차이 흡수)
        fliplr=0.5, flipud=0.0,
        degrees=5.0, translate=0.1, scale=0.5,
        mosaic=1.0, close_mosaic=10,          # 마지막 10 epoch은 mosaic 끄기
        copy_paste=0.3,                       # 희귀 dirt 인스턴스 복제-붙여넣기(④)

        val=True,
        plots=True,
    )
    print(f"\n학습 결과: {C.RUNS_DIR}/blade30_finetune  (best.pt = weights/best.pt)")


if __name__ == "__main__":
    main()
