import os, json, contextlib, io, sys, time
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IMG_DIR = "blade30_yolo_defects/images"
LBL_DIR = "blade30_yolo_defects/labels"
CATS = [{"id": 0, "name": "dirt"}, {"id": 1, "name": "damage"}]
CONF = 0.10  # low conf so mAP captures the full precision-recall curve

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---- build ground-truth COCO ----
img_files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
imgid = {f: i for i, f in enumerate(img_files)}
images, annos, ann_id = [], [], 1
for f in img_files:
    w, h = Image.open(os.path.join(IMG_DIR, f)).size
    images.append({"id": imgid[f], "file_name": f, "width": w, "height": h})
    lp = os.path.join(LBL_DIR, os.path.splitext(f)[0] + ".txt")
    if os.path.exists(lp):
        for line in open(lp):
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = map(float, p)
            x, y, ww, hh = (cx - bw/2)*w, (cy - bh/2)*h, bw*w, bh*h
            annos.append({"id": ann_id, "image_id": imgid[f], "category_id": int(c),
                          "bbox": [x, y, ww, hh], "area": ww*hh, "iscrowd": 0})
            ann_id += 1
json.dump({"images": images, "annotations": annos, "categories": CATS}, open("_gt.json", "w"))
log(f"GT built: {len(images)} images, {len(annos)} boxes")

def run(mode):
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path="best.pt",
        confidence_threshold=CONF, device="cpu")
    preds = []
    t0 = time.time()
    for k, f in enumerate(img_files, 1):
        p = os.path.join(IMG_DIR, f)
        if mode == "plain":
            r = get_prediction(p, model)
        else:
            r = get_sliced_prediction(
                p, model, slice_height=960, slice_width=960,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                postprocess_type="GREEDYNMM", postprocess_match_metric="IOS",
                postprocess_match_threshold=0.4, verbose=0)
        for o in r.object_prediction_list:
            b = o.bbox
            preds.append({"image_id": imgid[f], "category_id": int(o.category.id),
                          "bbox": [float(b.minx), float(b.miny),
                                   float(b.maxx - b.minx), float(b.maxy - b.miny)],
                          "score": float(o.score.value)})
        if mode == "sahi" and k % 10 == 0:
            el = time.time() - t0
            log(f"  sahi {k}/{len(img_files)}  ({el/k:.1f}s/img, eta {el/k*(len(img_files)-k)/60:.1f}min)")
    return preds

def evaluate(preds, tag):
    json.dump(preds, open(f"_pred_{tag}.json", "w"))
    coco = COCO("_gt.json")
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(f"_pred_{tag}.json") if preds else coco.loadRes([])
        e = COCOeval(coco, dt, "bbox"); e.evaluate(); e.accumulate(); e.summarize()
    log(f"RESULT {tag}: mAP50-95={e.stats[0]:.4f}  mAP50={e.stats[1]:.4f}  (#preds={len(preds)})")
    return e.stats[0], e.stats[1]

log("=== running PLAIN (whole-image) ===")
evaluate(run("plain"), "plain")
log("=== running SAHI (tuned: conf0.1, slice960, GNMM/IOS0.4) ===")
evaluate(run("sahi"), "sahi")
log("DONE")
