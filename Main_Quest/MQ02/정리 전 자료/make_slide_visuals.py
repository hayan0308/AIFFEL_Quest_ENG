"""
발표용 비교 그림 생성: 같은 blade30 이미지에서 [단일 챔피언] vs [앙상블] 을 나란히.
- GT(정답)=초록,  damage 예측=빨강,  dirt 예측=파랑
- GT 박스가 많은 대표 이미지 N장 자동 선택
출력: slide_visuals/<이미지>.png  (좌: 단일 챔피언 / 우: 앙상블)
"""
import os, glob
import numpy as np
import cv2
from ultralytics import YOLO, RTDETR
from ensemble_boxes import weighted_boxes_fusion

ROOT = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(ROOT, "ensemble_bundle", "weights")
IMG_DIR = os.path.join(ROOT, "blade30_yolo_defects", "images")
LBL_DIR = os.path.join(ROOT, "blade30_yolo_defects", "labels")
OUT_DIR = os.path.join(ROOT, "slide_visuals")
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES   = 5      # 뽑을 대표 이미지 수
DISP_CONF  = 0.25   # 화면에 그릴 최소 신뢰도
PANEL_W    = 1500   # 패널 한 개 가로 크기
WBF_IOU    = 0.55
RTDETR_W   = 0.5
NAMES = {0: "dirt", 1: "damage"}
COL_PRED = {0: (255, 128, 0), 1: (0, 0, 255)}   # dirt=파랑, damage=빨강 (BGR)
COL_GT   = (0, 200, 0)

YOLO_WEIGHTS = {
    "champ_yolov8s": "yolov8s_norm1500.pt",
    "yolo11s":  "yolo11s_norm750.pt",
    "yolov10s": "yolov10s_norm750.pt",
    "yolo12s":  "yolo12s_norm750.pt",
}
CHAMP = "champ_yolov8s"


def read_gt(lbl, W, H):
    out = []
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) != 5:
                continue
            c = int(float(p[0])); cx, cy, w, h = map(float, p[1:])
            out.append((c, (cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H))
    return out


def predict(model, path, W, H):
    r = model.predict(path, conf=DISP_CONF, verbose=False)[0]
    b, s, l = [], [], []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        b.append([x1/W, y1/H, x2/W, y2/H]); s.append(float(box.conf[0])); l.append(int(box.cls[0]))
    return b, s, l


def draw(img, boxes, gts):
    """boxes: [(cls,score,x1,y1,x2,y2)] 픽셀좌표. GT는 초록으로 먼저."""
    im = img.copy()
    for c, x1, y1, x2, y2 in gts:
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), COL_GT, 6)
    for c, sc, x1, y1, x2, y2 in boxes:
        col = COL_PRED.get(c, (0, 0, 255))
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), col, 4)
        txt = f"{NAMES.get(c,c)} {sc:.2f}"
        cv2.putText(im, txt, (int(x1), max(0, int(y1)-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 4, cv2.LINE_AA)
    return im


def banner(im, text):
    h = 70
    bar = np.zeros((h, im.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    return np.vstack([bar, im])


def main():
    imgs = sorted(p for p in glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    # GT 박스 수로 정렬해 상위 N장 선택 (담을 게 많은 대표 장면)
    scored = []
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        n = len(read_gt(os.path.join(LBL_DIR, stem + ".txt"), 1, 1))
        scored.append((n, p))
    scored.sort(reverse=True)
    picks = [p for _, p in scored[:N_IMAGES]]
    print("선택된 이미지:", [os.path.basename(p) for p in picks])

    print("[로드] 모델 5종")
    models = {n: YOLO(os.path.join(WDIR, f)) for n, f in YOLO_WEIGHTS.items()}
    rtdetr = RTDETR(os.path.join(WDIR, "rtdetr-l.pt"))

    for p in picks:
        stem = os.path.splitext(os.path.basename(p))[0]
        im = cv2.imread(p); H, W = im.shape[:2]
        gts = read_gt(os.path.join(LBL_DIR, stem + ".txt"), W, H)

        # 단일 챔피언
        cb, cs, cl = predict(models[CHAMP], p, W, H)
        champ_boxes = [(cl[i], cs[i], cb[i][0]*W, cb[i][1]*H, cb[i][2]*W, cb[i][3]*H)
                       for i in range(len(cb))]

        # 앙상블 (WBF, 5종)
        all_b, all_s, all_l, wts = [], [], [], []
        for n, m in models.items():
            b, s, l = predict(m, p, W, H); all_b.append(b); all_s.append(s); all_l.append(l); wts.append(1)
        b, s, l = predict(rtdetr, p, W, H); all_b.append(b); all_s.append(s); all_l.append(l); wts.append(RTDETR_W)
        fb, fs, fl = weighted_boxes_fusion(all_b, all_s, all_l, weights=wts, iou_thr=WBF_IOU, skip_box_thr=0.0)
        ens_boxes = [(int(fl[i]), float(fs[i]), fb[i][0]*W, fb[i][1]*H, fb[i][2]*W, fb[i][3]*H)
                     for i in range(len(fb)) if fs[i] >= DISP_CONF]

        left  = draw(im, champ_boxes, gts)
        right = draw(im, ens_boxes, gts)
        left  = banner(left,  f"Single champion (yolov8s): {len(champ_boxes)} det")
        right = banner(right, f"Ensemble WBF (5 models): {len(ens_boxes)} det")

        # 리사이즈 + 가로 결합 (가운데 흰 구분선)
        def rs(x):
            r = PANEL_W / x.shape[1]
            return cv2.resize(x, (PANEL_W, int(x.shape[0]*r)))
        left, right = rs(left), rs(right)
        gap = np.full((left.shape[0], 8, 3), 255, np.uint8)
        combo = np.hstack([left, gap, right])
        out = os.path.join(OUT_DIR, stem + ".png")
        cv2.imwrite(out, combo)
        print(f"  저장: {out}  (champ {len(champ_boxes)} vs ens {len(ens_boxes)}, GT {len(gts)})")

    print(f"\n완료. GT=초록, damage=빨강, dirt=파랑.  -> {OUT_DIR}")


if __name__ == "__main__":
    main()
