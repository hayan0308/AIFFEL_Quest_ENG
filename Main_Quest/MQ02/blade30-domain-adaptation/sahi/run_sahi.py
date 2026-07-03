import os, glob
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from PIL import Image

IMG_DIR = "blade30_yolo_defects/images"
OUT_DIR = "predict_runs/sahi_test"
os.makedirs(OUT_DIR, exist_ok=True)

CONF = 0.25
model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="best.pt",
    confidence_threshold=CONF,
    device="cpu",
)

# pick a representative subset (incl. ones that had 0 detections in plain run)
picks = [
    "Blade_5__19_dc11bd10-c1c6-4709-a4b9-491806ecf633.jpg",  # had detections
    "Blade_3__2_b8937f37-2983-477f-b5ca-40b9530512fb.jpg",   # no detections plain
    "Blade_5__7_18f1b84a-f655-4cca-87bc-d277fc87d1ba.jpg",   # no detections plain
    "Blade_9__4_e7123c5d-8801-41d0-a9fc-b4b97d594483.jpg",   # no detections plain
    "Blade_7__3_e983b71c-dad7-420e-931e-3d8faaf8f5c4.jpg",   # 7 damages plain
    "Blade_6__16_e48513e2-6614-4559-9325-f7fb4be3db52.jpg",  # no detections plain
]
picks = [p for p in picks if os.path.exists(os.path.join(IMG_DIR, p))]

print(f"{'image':<45} {'size':<12} {'plain':>6} {'SAHI':>6}")
print("-" * 75)
for name in picks:
    path = os.path.join(IMG_DIR, name)
    w, h = Image.open(path).size

    # plain (whole-image) prediction
    plain = get_prediction(path, model)
    n_plain = len(plain.object_prediction_list)

    # sliced prediction
    sliced = get_sliced_prediction(
        path, model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        verbose=0,
    )
    n_sahi = len(sliced.object_prediction_list)

    print(f"{name[:44]:<45} {f'{w}x{h}':<12} {n_plain:>6} {n_sahi:>6}")

    # save SAHI visualization
    sliced.export_visuals(export_dir=OUT_DIR, file_name=name.rsplit('.',1)[0])

print(f"\nSAHI visuals saved to: {OUT_DIR}")
