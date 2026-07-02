#!/usr/bin/env python3
"""
Blade30 (LabelMe polygon JSON) -> YOLO detection format 변환기

- 베이스 데이터(Kaggle NordTank586x371)와 동일한 2클래스 체계로 통합:
    0 = Dirt   (오염 / contamination)
    1 = Damage (손상 / erosion, crack)
- add-on(피뢰침, 마킹, serration 등 정상 부품)은 결함이 아니므로 제외
- 라벨(JSON) 없는 이미지는 '정상' 이미지로 간주 -> 빈 라벨(.txt)로 포함
  (Kaggle에서 라벨 없는 이미지 == 정상, 과 동일한 규칙)

표준 라이브러리만 사용 (외부 패키지 불필요). 이미지 크기는 JSON의
imageWidth/imageHeight 를 사용하고, JSON이 없는 이미지는 아래 옵션으로 처리.

사용법:
    python3 convert_blade30_to_yolo.py
    python3 convert_blade30_to_yolo.py --out blade30_yolo --copy   # 이미지 복사
    python3 convert_blade30_to_yolo.py --no-normal                 # 정상이미지 제외
"""
import argparse
import glob
import json
import os
import shutil

# ---- 클래스 정의 (Kaggle과 동일) ----
DIRT = 0
DAMAGE = 1
CLASS_NAMES = ["Dirt", "Damage"]


# add-on;LPS;worn or burnt(낙뢰보호장치 마모/소손)를 Damage로 볼지 여부
LPS_WORN_AS_DAMAGE = True

# Damage(1)로 매핑할 결함 키워드 (leading/trailing edge, surface 결함 전반)
DAMAGE_KEYWORDS = ["erosion", "crack", "coating damage",
                   "laminate defect", "burn damage"]


def map_label(label: str):
    """Blade30 라벨 문자열 -> Kaggle 클래스 id (제외 시 None)."""
    l = label.lower()
    if "contamination" in l:                       # ...;contamination;dirt / other -> 오염
        return DIRT
    if any(k in l for k in DAMAGE_KEYWORDS):        # erosion/crack/coating·burn damage 등 -> 손상
        return DAMAGE
    if "lps" in l and ("worn" in l or "burnt" in l):  # 낙뢰보호장치 마모/소손
        return DAMAGE if LPS_WORN_AS_DAMAGE else None
    return None                                     # add-on;*;OK, markings 등 -> 제외


def poly_to_yolo(points, w, h):
    """폴리곤 꼭짓점 -> YOLO (xc, yc, bw, bh) 정규화 좌표. 유효하지 않으면 None."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = max(0.0, min(xs)), min(float(w), max(xs))
    ymin, ymax = max(0.0, min(ys)), min(float(h), max(ys))
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 1 or bh <= 1:                    # 너무 작거나 뒤집힌 박스 무시
        return None
    xc = (xmin + xmax) / 2.0 / w
    yc = (ymin + ymax) / 2.0 / h
    return xc, yc, bw / w, bh / h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".", help="Blade_* 폴더들이 있는 루트")
    ap.add_argument("--out", default="blade30_yolo", help="출력 폴더")
    ap.add_argument("--copy", action="store_true",
                    help="이미지를 복사 (기본은 심볼릭 링크)")
    ap.add_argument("--no-normal", action="store_true",
                    help="라벨 없는 정상 이미지를 결과에서 제외")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    lbl_dir = os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    blade_dirs = sorted(glob.glob(os.path.join(args.src, "Blade_*")))
    all_jpgs = []
    for bd in blade_dirs:
        all_jpgs += glob.glob(os.path.join(bd, "**", "*.jpg"), recursive=True)
    all_jpgs = [p for p in all_jpgs if os.sep + "mask" + os.sep not in p]

    stats = {"images": 0, "with_defect": 0, "normal": 0,
             "boxes": {DIRT: 0, DAMAGE: 0}, "skipped_addon": 0, "no_size": 0}

    for jpg in sorted(all_jpgs):
        blade = jpg.split(os.sep)[len(args.src.rstrip(os.sep).split(os.sep))]  # Blade_N
        base = os.path.splitext(os.path.basename(jpg))[0]
        stem = f"{blade}__{base}"                      # 블레이드 출처 보존 + 충돌 방지
        jpath = jpg[:-4] + ".json"

        lines = []
        if os.path.exists(jpath):
            with open(jpath) as f:
                data = json.load(f)
            w, h = data.get("imageWidth"), data.get("imageHeight")
            if not w or not h:
                stats["no_size"] += 1
                continue
            for shp in data.get("shapes", []):
                cid = map_label(shp.get("label", ""))
                if cid is None:
                    stats["skipped_addon"] += 1
                    continue
                box = poly_to_yolo(shp["points"], w, h)
                if box is None:
                    continue
                lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in box))
                stats["boxes"][cid] += 1
        else:
            # JSON 없음 = 정상 이미지
            if args.no_normal:
                continue

        # 결함이 하나도 없으면 정상 이미지(빈 라벨)
        if lines:
            stats["with_defect"] += 1
        else:
            stats["normal"] += 1
        stats["images"] += 1

        # 라벨 파일 (정상 이미지는 빈 파일)
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        # 이미지 배치 (복사 or 심볼릭 링크)
        dst = os.path.join(img_dir, stem + ".jpg")
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        if args.copy:
            shutil.copy2(jpg, dst)
        else:
            os.symlink(os.path.abspath(jpg), dst)

    # classes.txt / data.yaml 조각 출력
    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        f.write("\n".join(CLASS_NAMES) + "\n")

    print("=== 변환 완료 ===")
    print(f"출력 폴더        : {args.out}/  (images/, labels/)")
    print(f"총 이미지        : {stats['images']}")
    print(f"  - 결함 포함     : {stats['with_defect']}")
    print(f"  - 정상(빈 라벨) : {stats['normal']}")
    print(f"박스 수          : Dirt={stats['boxes'][DIRT]}, Damage={stats['boxes'][DAMAGE]}")
    print(f"제외된 add-on    : {stats['skipped_addon']}")
    if stats["no_size"]:
        print(f"크기정보 없음(스킵): {stats['no_size']}")


if __name__ == "__main__":
    main()
