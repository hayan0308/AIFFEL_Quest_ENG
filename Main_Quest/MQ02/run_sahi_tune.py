import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

IMG_DIR = "blade30_yolo_defects/images"
OUT_DIR = "predict_runs/sahi_tuned"
os.makedirs(OUT_DIR, exist_ok=True)

picks = [
    "Blade_5__19_dc11bd10-c1c6-4709-a4b9-491806ecf633.jpg",  # over-detection case (157)
    "Blade_5__7_18f1b84a-f655-4cca-87bc-d277fc87d1ba.jpg",   # good recovery case (16)
    "Blade_7__3_e983b71c-dad7-420e-931e-3d8faaf8f5c4.jpg",   # partial (32)
    "Blade_6__16_e48513e2-6614-4559-9325-f7fb4be3db52.jpg",  # recovery (9)
]
picks = [p for p in picks if os.path.exists(os.path.join(IMG_DIR, p))]

# (label, conf, slice, postprocess_type, match_metric, match_thr)
configs = [
    ("baseline  c25 s640 GNMM/IOS.5", 0.25, 640, "GREEDYNMM", "IOS", 0.5),
    ("A c50 s640 NMS/IOU.5",          0.50, 640, "NMS",       "IOU", 0.5),
    ("B c50 s960 NMS/IOU.5",          0.50, 960, "NMS",       "IOU", 0.5),
    ("C c50 s960 GNMM/IOS.4",         0.50, 960, "GREEDYNMM", "IOS", 0.4),
]

# cache one model per conf level
models = {}
def get_model(conf):
    if conf not in models:
        models[conf] = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path="best.pt",
            confidence_threshold=conf, device="cpu")
    return models[conf]

# header
print(f"{'image':<24}" + "".join(f"{c[0].split()[0]+c[0].split()[1]:>16}" for c in configs))
results = {}
for name in picks:
    path = os.path.join(IMG_DIR, name)
    row = []
    for label, conf, sl, pptype, metric, thr in configs:
        r = get_sliced_prediction(
            path, get_model(conf),
            slice_height=sl, slice_width=sl,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
            postprocess_type=pptype, postprocess_match_metric=metric,
            postprocess_match_threshold=thr, verbose=0)
        n = len(r.object_prediction_list)
        row.append(n)
        results[(name, label)] = r
    short = name.split("_")[0] + "_" + name.split("__")[0].split("_")[-1] + "_" + name.split("__")[1].split("_")[0]
    print(f"{short:<24}" + "".join(f"{x:>16}" for x in row))

# export visuals for config C on the two key images
for name in [picks[0], picks[1]]:
    r = results[(name, "C c50 s960 GNMM/IOS.4")]
    r.export_visuals(export_dir=OUT_DIR, file_name="C_" + name.rsplit('.',1)[0])
print(f"\nConfig C visuals saved to: {OUT_DIR}")
