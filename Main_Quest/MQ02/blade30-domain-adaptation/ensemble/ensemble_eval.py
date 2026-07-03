"""
MQ02 WBF 앙상블 '평가기' — 라벨(GT) 있는 새 데이터에서 앙상블 점수를 잰다 (자체 완결형)

★ 언제 쓰나:
  팀원이 라벨링을 끝낸 데이터가 있을 때, "우리 앙상블이 그 데이터에서
  mAP50 / damage recall 이 얼마나 나오나" 를 측정한다. 단일 챔피언(yolov8s)과도 비교.

★ 필요한 데이터 구조 (YOLO 포맷):
    <DATA_DIR>/
      images/  <이름>.png (또는 jpg)
      labels/  <이름>.txt   ← "class cx cy w h" (정규화), 이미지와 파일명 짝
  클래스 번호는 우리와 같아야 함: 0=dirt, 1=damage.
  (팀원 데이터가 손상을 crack/erosion 등으로 세분화했다면, 그걸 전부 1(damage)로,
   오염류는 0(dirt)으로 remap 한 라벨을 넣어야 비교가 맞다. README 참고.)

── 사용법 ──────────────────────────────────────────
  1) pip install ultralytics ensemble-boxes opencv-python
  2) 아래 [설정] DATA_DIR 을 팀원 데이터 폴더로
  3) python ensemble_eval.py
출력: 단일 4종 + 앙상블(YOLO4 / +rtdetr) 의 mAP50 / damage recall / precision 표
     (+ SAVE_VIS=True 면 output/vis 에 앙상블 박스 그림도 저장)
──────────────────────────────────────────────────
지표 주의: AP는 conf>=PRED_CONF(0.05)로 계산 = ultralytics 공식 절대값과 다를 수 있음.
          단일·앙상블에 같은 잣대라 '상대비교(앙상블이 오르나)'는 유효.
"""
import os, glob, json
import numpy as np
import cv2
from ultralytics import YOLO, RTDETR
from ensemble_boxes import weighted_boxes_fusion

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(HERE, "weights")

# ================== [설정] ==================
DATA_DIR  = os.path.join(HERE, "input")     # ★ images/ 와 labels/ 를 가진 폴더
OUT_DIR   = os.path.join(HERE, "output")
PRED_CONF = 0.05
WBF_IOU   = 0.55
RTDETR_WEIGHT = 0.5
USE_RTDETR = True
SAVE_VIS  = False    # True 면 앙상블 박스 그림도 저장(느려짐)
NAMES = {0: "dirt", 1: "damage"}
# ===========================================

YOLO_WEIGHTS = {
    "yolov8s_n1500": "yolov8s_norm1500.pt",   # 단일 챔피언
    "yolo11s_n750":  "yolo11s_norm750.pt",
    "yolov10s_n750": "yolov10s_norm750.pt",
    "yolo12s_n750":  "yolo12s_norm750.pt",
}
CHAMP = "yolov8s_n1500"
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


# ── GT/지표 계산 (외부 의존 없이 자체 포함) ──────────────────
def read_gt(lbl_path, W, H):
    out = []
    if os.path.exists(lbl_path):
        for line in open(lbl_path):
            p = line.split()
            if len(p) != 5:
                continue
            c = int(p[0]); cx, cy, w, h = map(float, p[1:])
            out.append((c, (cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H))
    return out


def iou(a, b):
    b = np.asarray(b).reshape(-1, 4)
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2-x1, 0, None) * np.clip(y2-y1, 0, None)
    area_a = (a[2]-a[0])*(a[3]-a[1]); area_b = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def ap_recall(preds, gts, iou_thr=0.5):
    """preds: [(img,score,box)], gts: {img:[box]} (한 클래스만) -> AP50, recall, precision."""
    n_gt = sum(len(v) for v in gts.values())
    if n_gt == 0:
        return float("nan"), float("nan"), float("nan")
    preds = sorted(preds, key=lambda x: -x[1])
    matched = {img: np.zeros(len(gts.get(img, [])), bool) for img in gts}
    tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
    for i, (img, sc, box) in enumerate(preds):
        gt = gts.get(img, [])
        if len(gt) == 0:
            fp[i] = 1; continue
        ious = iou(np.asarray(box), gt); j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not matched[img][j]:
            tp[i] = 1; matched[img][j] = True
        else:
            fp[i] = 1
    ctp = np.cumsum(tp); cfp = np.cumsum(fp)
    rec = ctp / n_gt; prec = ctp / (ctp + cfp + 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for k in range(len(mpre)-1, 0, -1):
        mpre[k-1] = max(mpre[k-1], mpre[k])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx+1]-mrec[idx]) * mpre[idx+1]))
    return ap, float(rec[-1] if len(rec) else 0), float(prec[-1] if len(prec) else 0)


def predict_norm(model, img_path, W, H):
    r = model.predict(img_path, conf=PRED_CONF, verbose=False)[0]
    b, s, l = [], [], []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        b.append([x1/W, y1/H, x2/W, y2/H]); s.append(float(box.conf[0])); l.append(int(box.cls[0]))
    return b, s, l


def main():
    img_dir = os.path.join(DATA_DIR, "images")
    lbl_dir = os.path.join(DATA_DIR, "labels")
    imgs = sorted(p for p in glob.glob(os.path.join(img_dir, "*")) if p.lower().endswith(IMG_EXT))
    if not imgs:
        print(f"[에러] {img_dir} 에 이미지가 없습니다. DATA_DIR 구조(images/ labels/)를 확인하세요.")
        return
    print(f"[로드] YOLO 4종{' + rtdetr' if USE_RTDETR else ''}")
    models = {n: YOLO(os.path.join(WDIR, f)) for n, f in YOLO_WEIGHTS.items()}
    rtdetr = RTDETR(os.path.join(WDIR, "rtdetr-l.pt")) if USE_RTDETR else None
    methods = list(YOLO_WEIGHTS) + (["ENS_YOLO", "ENS_ALL(+rtdetr)"] if USE_RTDETR else ["ENS_YOLO"])
    preds = {m: {0: [], 1: []} for m in methods}
    gts = {0: {}, 1: {}}
    if SAVE_VIS:
        os.makedirs(os.path.join(OUT_DIR, "vis"), exist_ok=True)

    print(f"[평가] {len(imgs)}장")
    for k, ip in enumerate(imgs):
        stem = os.path.splitext(os.path.basename(ip))[0]
        im = cv2.imread(ip); H, W = im.shape[:2]
        for c, x1, y1, x2, y2 in read_gt(os.path.join(lbl_dir, stem + ".txt"), W, H):
            if c in gts:
                gts[c].setdefault(stem, []).append([x1, y1, x2, y2])

        all_b, all_s, all_l, wts = [], [], [], []
        for n, m in models.items():
            b, s, l = predict_norm(m, ip, W, H)
            for (bx, by, bx2, by2), sc, cl in zip(b, s, l):
                preds[n][cl].append((stem, sc, [bx*W, by*H, bx2*W, by2*H]))
            all_b.append(b); all_s.append(s); all_l.append(l); wts.append(1)
        if rtdetr is not None:
            b, s, l = predict_norm(rtdetr, ip, W, H)
            all_b.append(b); all_s.append(s); all_l.append(l); wts.append(RTDETR_WEIGHT)

        # ENS_YOLO (rtdetr 제외분)
        yb, ys, yl = weighted_boxes_fusion(all_b[:4], all_s[:4], all_l[:4],
                                           weights=[1]*4, iou_thr=WBF_IOU, skip_box_thr=0.0)
        for (bx, by, bx2, by2), sc, cl in zip(yb, ys, yl):
            preds["ENS_YOLO"][int(cl)].append((stem, float(sc), [bx*W, by*H, bx2*W, by2*H]))
        # ENS_ALL (+rtdetr)
        if USE_RTDETR:
            fb, fs, fl = weighted_boxes_fusion(all_b, all_s, all_l, weights=wts,
                                               iou_thr=WBF_IOU, skip_box_thr=0.0)
            for (bx, by, bx2, by2), sc, cl in zip(fb, fs, fl):
                preds["ENS_ALL(+rtdetr)"][int(cl)].append((stem, float(sc), [bx*W, by*H, bx2*W, by2*H]))
                if SAVE_VIS and float(sc) >= 0.25:
                    x1, y1, x2, y2 = int(bx*W), int(by*H), int(bx2*W), int(by2*H)
                    col = (0, 0, 255) if int(cl) == 1 else (0, 180, 0)
                    cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
            if SAVE_VIS:
                cv2.imwrite(os.path.join(OUT_DIR, "vis", stem + ".png"), im)
        if (k+1) % 20 == 0:
            print(f"  ...{k+1}/{len(imgs)}")

    # ── 결과표 (화면 + 파일 저장) ──
    n_gt0 = sum(len(v) for v in gts[0].values())
    n_gt1 = sum(len(v) for v in gts[1].values())
    header = f"{'method':17} | {'mAP50':>7} | {'dirt AP':>7} {'dmg AP':>7} | {'dmg R':>7} {'dmg P':>7}"
    rows = []
    summary = {"n_images": len(imgs), "n_gt_dirt": n_gt0, "n_gt_damage": n_gt1,
               "use_rtdetr": USE_RTDETR, "methods": {}}
    for m in methods:
        ap0, r0, p0 = ap_recall(preds[m][0], gts[0])
        ap1, r1, p1 = ap_recall(preds[m][1], gts[1])
        mAP = float(np.nanmean([ap0, ap1]))
        tag = " <--" if m.startswith("ENS") else (" (챔피언)" if m == CHAMP else "")
        rows.append(f"{m:17} | {mAP:7.4f} | {ap0:7.4f} {ap1:7.4f} | {r1:7.4f} {p1:7.4f}{tag}")
        summary["methods"][m] = dict(mAP50=mAP, dirtAP=ap0, dmgAP=ap1, dmgR=r1, dmgP=p1)

    lines = ["="*70,
             f"MQ02 앙상블 평가 결과  (이미지 {len(imgs)}장, GT dirt {n_gt0} / damage {n_gt1})",
             "="*70, header, "-"*70, *rows, "="*70,
             "관전: 앙상블(ENS) dmg R 이 단일 챔피언보다 오르나 / 이 새 데이터에서도 그런가"]
    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {OUT_DIR}/results.txt (표) + summary.json (수치)")
    print("★ 이 두 파일을 김민욱에게 보내면 됩니다. (SAVE_VIS=True 였다면 output/vis 몇 장도 함께)")


if __name__ == "__main__":
    main()
