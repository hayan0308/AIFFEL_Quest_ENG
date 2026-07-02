import os, random, math
from PIL import Image

random.seed(42)

SRC_IMG = "blade30_yolo_defects/images"
SRC_LBL = "blade30_yolo_defects/labels"
OUT = "blade30_tiles"
TILE = 960
OVERLAP = 0.2
STEP = int(TILE * (1 - OVERLAP))       # 768
MIN_VIS = 0.25                          # keep clipped box if >=25% of original area survives
MIN_PX = 4                              # and both sides >= 4 px
VAL_RATIO = 0.20
NEG_RATIO = 0.20                        # empty tiles kept, per image, as fraction of positive tiles
NAMES = {0: "dirt", 1: "damage"}

for split in ("train", "val"):
    os.makedirs(f"{OUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)

img_files = sorted(f for f in os.listdir(SRC_IMG) if f.lower().endswith((".jpg", ".jpeg", ".png")))
random.shuffle(img_files)
n_val = int(len(img_files) * VAL_RATIO)
split_of = {f: ("val" if i < n_val else "train") for i, f in enumerate(img_files)}

def tile_origins(size):
    if size <= TILE:
        return [0]
    xs = list(range(0, size - TILE + 1, STEP))
    if xs[-1] != size - TILE:
        xs.append(size - TILE)          # clamp last tile to fit
    return xs

stats = {"train": {"pos": 0, "neg": 0, "boxes": {0: 0, 1: 0}},
         "val":   {"pos": 0, "neg": 0, "boxes": {0: 0, 1: 0}}}

for f in img_files:
    split = split_of[f]
    im = Image.open(os.path.join(SRC_IMG, f)).convert("RGB")
    W, H = im.size
    stem = os.path.splitext(f)[0]

    # load GT boxes in absolute pixel corners
    boxes = []  # (cls, x1, y1, x2, y2)
    lp = os.path.join(SRC_LBL, stem + ".txt")
    if os.path.exists(lp):
        for line in open(lp):
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = float(p[0]), *map(float, p[1:])
            x1, y1 = (cx - bw/2) * W, (cy - bh/2) * H
            x2, y2 = (cx + bw/2) * W, (cy + bh/2) * H
            boxes.append((int(c), x1, y1, x2, y2))

    pos_tiles, neg_tiles = [], []
    for r, y0 in enumerate(tile_origins(H)):
        for cidx, x0 in enumerate(tile_origins(W)):
            tx2, ty2 = x0 + TILE, y0 + TILE
            labels = []
            for cls, bx1, by1, bx2, by2 in boxes:
                ix1, iy1 = max(bx1, x0), max(by1, y0)
                ix2, iy2 = min(bx2, tx2), min(by2, ty2)
                iw, ih = ix2 - ix1, iy2 - iy1
                if iw <= 0 or ih <= 0:
                    continue
                orig_area = (bx2 - bx1) * (by2 - by1)
                if orig_area <= 0:
                    continue
                if (iw * ih) / orig_area < MIN_VIS or iw < MIN_PX or ih < MIN_PX:
                    continue
                # to tile-normalized xywh
                ncx = ((ix1 + ix2) / 2 - x0) / TILE
                ncy = ((iy1 + iy2) / 2 - y0) / TILE
                nw, nh = iw / TILE, ih / TILE
                labels.append((cls, ncx, ncy, nw, nh))
            entry = (r, cidx, x0, y0, labels)
            (pos_tiles if labels else neg_tiles).append(entry)

    # keep all positive tiles + capped random negatives
    keep_neg = min(len(neg_tiles), math.ceil(NEG_RATIO * len(pos_tiles)))
    random.shuffle(neg_tiles)
    kept = pos_tiles + neg_tiles[:keep_neg]

    for r, cidx, x0, y0, labels in kept:
        crop = im.crop((x0, y0, x0 + TILE, y0 + TILE))
        tname = f"{stem}_r{r}_c{cidx}"
        crop.save(f"{OUT}/images/{split}/{tname}.jpg", quality=95)
        with open(f"{OUT}/labels/{split}/{tname}.txt", "w") as w:
            for cls, ncx, ncy, nw, nh in labels:
                w.write(f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")
                stats[split]["boxes"][cls] += 1
        stats[split]["pos" if labels else "neg"] += 1

with open(f"{OUT}/data.yaml", "w") as w:
    w.write(f"path: {os.path.abspath(OUT)}\n")
    w.write("train: images/train\nval: images/val\n")
    w.write("names:\n  0: dirt\n  1: damage\n")

print(f"tiles: {TILE}px, step {STEP}px, val_ratio {VAL_RATIO}")
for s in ("train", "val"):
    st = stats[s]
    print(f"{s:5}: {st['pos']} positive + {st['neg']} background tiles | "
          f"boxes dirt={st['boxes'][0]}, damage={st['boxes'][1]}")
print(f"data.yaml -> {OUT}/data.yaml")
