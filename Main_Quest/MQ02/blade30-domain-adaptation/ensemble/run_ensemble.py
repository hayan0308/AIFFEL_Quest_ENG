"""
MQ02 WBF 앙상블 추론기 — 새 이미지 폴더에 그대로 적용 (정답 라벨 불필요)

★ 이게 뭐냐:
  우리가 학습한 5개 모델(YOLO 4종 + rtdetr)의 예측을 WBF(Weighted Boxes Fusion)로 합쳐
  손상(damage)/오염(dirt)을 하나의 박스로 내놓는다. 학습 안 함 = 추론만. GT 없어도 됨.
  그래서 라벨 안 된 '새 데이터'에 바로 붙일 수 있다.

★ 왜 앙상블인가:
  단일 최고 모델보다 damage recall(놓침)이 확실히 낮아진다(우리 test에서 0.856 -> 0.944).
  서로 다른 모델이 놓친 걸 메워주기 때문.

── 사용법 (3단계) ────────────────────────────────────────────
  1) 패키지 설치:  pip install ultralytics ensemble-boxes opencv-python
  2) 아래 [설정]의 INPUT_DIR 을 '새 이미지 폴더' 경로로 바꾼다 (또는 input/ 에 이미지를 넣는다)
  3) python run_ensemble.py
결과:
  output/vis/*.png     — 박스 그려진 이미지 (damage=빨강, dirt=초록, 신뢰도 숫자)
  output/labels/*.txt  — YOLO 포맷 예측 라벨 (class cx cy w h, 정규화)
──────────────────────────────────────────────────────────
"""
import os, glob
import numpy as np
import cv2
from ultralytics import YOLO, RTDETR
from ensemble_boxes import weighted_boxes_fusion

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(HERE, "weights")

# ================== [설정] — 여기만 바꾸면 됨 ==================
INPUT_DIR = os.path.join(HERE, "input")     # ★ 새 이미지 폴더 경로로 바꾸세요
OUT_DIR   = os.path.join(HERE, "output")
OUT_CONF  = 0.25     # 최종 출력에 남길 신뢰도 하한(실사용 기준. 낮추면 더 많이, 지저분)
PRED_CONF = 0.05     # 각 모델이 후보를 넓게 내도록 낮게(WBF가 합치며 정리) — 건드리지 말 것
WBF_IOU   = 0.55     # 겹치는 박스끼리 묶는 IoU
RTDETR_WEIGHT = 0.5  # rtdetr은 반쪽학습이라 YOLO(1)보다 낮게 거들기만. rtdetr 빼려면 USE_RTDETR=False
USE_RTDETR = True    # False로 하면 YOLO 4종만(가볍고 빠름, recall 살짝 낮음 0.944->0.891)
# ============================================================

NAMES  = {0: "dirt", 1: "damage"}
COLORS = {0: (0, 180, 0), 1: (0, 0, 255)}   # BGR: dirt=초록, damage=빨강
YOLO_WEIGHTS = {   # 이름 -> weights/ 안 파일
    "yolov8s":  "yolov8s_norm1500.pt",
    "yolo11s":  "yolo11s_norm750.pt",
    "yolov10s": "yolov10s_norm750.pt",
    "yolo12s":  "yolo12s_norm750.pt",
}
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def predict_norm(model, img_path):
    """한 모델 예측 -> WBF용 정규화 박스(0~1), 점수, 라벨 리스트."""
    r = model.predict(img_path, conf=PRED_CONF, verbose=False)[0]
    W, H = r.orig_shape[1], r.orig_shape[0]
    boxes, scores, labels = [], [], []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        boxes.append([x1/W, y1/H, x2/W, y2/H])
        scores.append(float(b.conf[0])); labels.append(int(b.cls[0]))
    return boxes, scores, labels


def main():
    imgs = sorted(p for p in glob.glob(os.path.join(INPUT_DIR, "*"))
                  if p.lower().endswith(IMG_EXT))
    if not imgs:
        print(f"[에러] INPUT_DIR 에 이미지가 없습니다: {INPUT_DIR}")
        print("      -> input/ 폴더에 이미지를 넣거나, [설정]의 INPUT_DIR 을 바꾸세요.")
        return

    os.makedirs(os.path.join(OUT_DIR, "vis"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels"), exist_ok=True)

    print(f"[로드] YOLO 4종{' + rtdetr' if USE_RTDETR else ''} (weights/)")
    models = {n: YOLO(os.path.join(WDIR, f)) for n, f in YOLO_WEIGHTS.items()}
    rtdetr = RTDETR(os.path.join(WDIR, "rtdetr-l.pt")) if USE_RTDETR else None
    print(f"[추론] {len(imgs)}장 시작 (출력 신뢰도 >= {OUT_CONF})")

    for k, ip in enumerate(imgs):
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        all_b, all_s, all_l, wts = [], [], [], []
        for n, m in models.items():                 # YOLO 4종
            b, s, l = predict_norm(m, ip)
            all_b.append(b); all_s.append(s); all_l.append(l); wts.append(1)
        if rtdetr is not None:                       # rtdetr(가중치 낮게)
            b, s, l = predict_norm(rtdetr, ip)
            all_b.append(b); all_s.append(s); all_l.append(l); wts.append(RTDETR_WEIGHT)

        fb, fs, fl = weighted_boxes_fusion(all_b, all_s, all_l, weights=wts,
                                           iou_thr=WBF_IOU, skip_box_thr=0.0)

        # 결과 그리기 + 라벨 저장
        lines = []
        for (bx, by, bx2, by2), sc, cl in zip(fb, fs, fl):
            if sc < OUT_CONF:            # 신뢰도 낮은 건 최종 출력에서 제외
                continue
            cl = int(cl)
            x1, y1, x2, y2 = int(bx*W), int(by*H), int(bx2*W), int(by2*H)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[cl], 2)
            cv2.putText(img, f"{NAMES[cl]} {sc:.2f}", (x1, max(0, y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cl], 1, cv2.LINE_AA)
            cx, cy, w, h = (x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H
            lines.append(f"{cl} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        stem = os.path.splitext(os.path.basename(ip))[0]
        cv2.imwrite(os.path.join(OUT_DIR, "vis", stem + ".png"), img)
        with open(os.path.join(OUT_DIR, "labels", stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        if (k+1) % 20 == 0:
            print(f"  ...{k+1}/{len(imgs)}")

    print(f"[완료] 결과: {OUT_DIR}/vis (그림), {OUT_DIR}/labels (예측 라벨)")


if __name__ == "__main__":
    main()
