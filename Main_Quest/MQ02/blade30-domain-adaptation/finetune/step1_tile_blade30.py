"""
전처리 ① : 고해상도 blade30 이미지를 학습 해상도(TILE) 타일로 슬라이스한다.

- 스케일 불일치 해결: 결함이 모델이 익숙한 크기로 보이게 함 (학습용 SAHI)
- 109장 -> 수백~수천 개의 '실제 픽셀' 타일로 확장 (합성 증강이 아님)
- 결함이 없는 빈 타일 일부를 배경(negative)으로 보존 -> target 도메인 오탐 억제
- blade '그룹' 단위로 train/val 분할하여 같은 원본의 타일이 train/val에 섞이지 않게 함(누수 방지)

출력: finetune/dataset/blade30_tiles640/{train,val}/{images,labels}/
"""
import os, glob, random, shutil
from PIL import Image
import config as C


def load_labels(txt_path):
    """YOLO 라벨(cls cx cy w h, 정규화) 읽기."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path) as f:
        for line in f:
            p = line.split()
            if len(p) != 5:
                continue
            try:
                cls = int(float(p[0]))
            except ValueError:
                continue  # 손상된 라벨 라인 skip
            cx, cy, w, h = map(float, p[1:])
            boxes.append((cls, cx, cy, w, h))
    return boxes


def tile_positions(dim, tile, overlap):
    """한 축에 대한 타일 시작좌표 목록. 마지막 타일은 가장자리에 정렬."""
    step = int(tile * (1 - overlap))
    if dim <= tile:
        return [0]
    pos = list(range(0, dim - tile + 1, step))
    if pos[-1] != dim - tile:
        pos.append(dim - tile)
    return pos


def slice_image(img_path, lbl_path):
    """이미지 하나를 타일 목록으로 변환. 반환: [(crop_box, [labels]), ...]"""
    im = Image.open(img_path)
    W, H = im.size
    boxes_px = []
    for cls, cx, cy, w, h in load_labels(lbl_path):
        bw, bh = w * W, h * H
        x1, y1 = cx * W - bw / 2, cy * H - bh / 2
        boxes_px.append((cls, x1, y1, x1 + bw, y1 + bh))

    tiles = []
    for ty in tile_positions(H, C.TILE, C.OVERLAP):
        for tx in tile_positions(W, C.TILE, C.OVERLAP):
            tw = min(C.TILE, W - tx)
            th = min(C.TILE, H - ty)
            tile_boxes = []
            for cls, x1, y1, x2, y2 in boxes_px:
                ix1, iy1 = max(x1, tx), max(y1, ty)
                ix2, iy2 = min(x2, tx + tw), min(y2, ty + th)
                iw, ih = ix2 - ix1, iy2 - iy1
                if iw <= 0 or ih <= 0:
                    continue
                orig_area = (x2 - x1) * (y2 - y1)
                if orig_area <= 0 or (iw * ih) / orig_area < C.MIN_VIS:
                    continue  # 타일 안에 너무 조금 남으면 버림
                # 타일 로컬 좌표로 정규화
                ncx = (ix1 + ix2) / 2 - tx
                ncy = (iy1 + iy2) / 2 - ty
                tile_boxes.append((cls, ncx / tw, ncy / th, iw / tw, ih / th))
            tiles.append(((tx, ty, tx + tw, ty + th), tile_boxes))
    return im, tiles


def main():
    random.seed(C.SEED)
    img_dir = os.path.join(C.BLADE30_DIR, "images")
    lbl_dir = os.path.join(C.BLADE30_DIR, "labels")
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    # blade 그룹키(예: 'Blade_5')로 묶어 그룹 단위 분할 -> 누수 방지
    def group_key(path):
        base = os.path.basename(path)
        return "_".join(base.split("__")[0].split("_")[:2])  # 'Blade_5'

    groups = {}
    for p in images:
        groups.setdefault(group_key(p), []).append(p)
    keys = sorted(groups)
    random.shuffle(keys)
    n_val = max(1, int(len(keys) * C.VAL_FRACTION))
    val_keys = set(keys[:n_val])
    print(f"groups={len(keys)}  val_groups={len(val_keys)}  images={len(images)}")

    # 기존 출력 정리
    if os.path.exists(C.TILES_DIR):
        shutil.rmtree(C.TILES_DIR)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(C.TILES_DIR, split, sub), exist_ok=True)

    stats = {"train": [0, 0, 0], "val": [0, 0, 0]}  # [pos, neg_kept, boxes]
    for p in images:
        split = "val" if group_key(p) in val_keys else "train"
        lbl_path = os.path.join(lbl_dir, os.path.splitext(os.path.basename(p))[0] + ".txt")
        im, tiles = slice_image(p, lbl_path)
        stem = os.path.splitext(os.path.basename(p))[0]

        pos = [(box, lbs) for box, lbs in tiles if lbs]
        neg = [(box, lbs) for box, lbs in tiles if not lbs]
        # 배경 타일은 양성 수 대비 NEG_RATIO 만큼만 무작위 유지
        random.shuffle(neg)
        neg = neg[: int(len(pos) * C.NEG_RATIO)]

        for i, (box, lbs) in enumerate(pos + neg):
            crop = im.crop(box).convert("RGB")
            out_img = os.path.join(C.TILES_DIR, split, "images", f"{stem}_t{i}.jpg")
            out_lbl = os.path.join(C.TILES_DIR, split, "labels", f"{stem}_t{i}.txt")
            crop.save(out_img, quality=95)
            with open(out_lbl, "w") as f:
                for cls, cx, cy, w, h in lbs:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            stats[split][0 if lbs else 1] += 1
            stats[split][2] += len(lbs)

    for split in ("train", "val"):
        pos, neg, nb = stats[split]
        print(f"[{split}] positive_tiles={pos}  background_tiles={neg}  boxes={nb}")
    print(f"\n타일 저장 위치: {C.TILES_DIR}")


if __name__ == "__main__":
    main()
