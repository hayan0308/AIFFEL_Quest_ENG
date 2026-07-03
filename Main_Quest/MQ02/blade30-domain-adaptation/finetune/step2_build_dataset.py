"""
데이터 구성 ②③ : 최종 학습 데이터셋(train.txt / val.txt / finetune.yaml)을 만든다.

train = [ blade30 타일(양성+배경)  +  NordTank 라벨 이미지(망각 방지)  +  NordTank 정상 이미지(배경, 오탐 억제) ]
val   = [ blade30 타일만 ]   <- 우리가 실제로 올리고 싶은 target 도메인 지표

파일을 복사하지 않고 절대경로 목록(txt)으로 구성한다. (YOLO는 /images/ -> /labels/ 규칙으로 라벨을 찾음)
"""
import os, glob, random
import config as C


def main():
    random.seed(C.SEED)
    os.makedirs(os.path.dirname(C.DATA_YAML), exist_ok=True)

    # --- blade30 타일 ---
    b30_train = sorted(glob.glob(os.path.join(C.TILES_DIR, "train", "images", "*.jpg")))
    b30_val   = sorted(glob.glob(os.path.join(C.TILES_DIR, "val",   "images", "*.jpg")))
    if not b30_train:
        raise SystemExit("blade30 타일이 없습니다. 먼저 step1_tile_blade30.py 를 실행하세요.")

    # --- NordTank 라벨 이미지 (망각 방지용) ---
    nt_img_dir = os.path.join(C.NORDTANK_DIR, "images")
    nt_lbl_dir = os.path.join(C.NORDTANK_DIR, "labels")
    labeled, background = [], []
    for img in sorted(glob.glob(os.path.join(nt_img_dir, "*.png"))):
        stem = os.path.splitext(os.path.basename(img))[0]
        lbl = os.path.join(nt_lbl_dir, stem + ".txt")
        if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
            labeled.append(img)
        else:
            background.append(img)   # 라벨 없는 정상 이미지 = 배경(negative)

    random.shuffle(labeled)
    random.shuffle(background)
    nt_labeled = labeled[: C.NORDTANK_LABELED_MAX]
    nt_bg      = background[: C.NORDTANK_BG_MAX]

    train_list = b30_train + nt_labeled + nt_bg
    val_list   = b30_val
    random.shuffle(train_list)

    train_txt = os.path.join(os.path.dirname(C.DATA_YAML), "train.txt")
    val_txt   = os.path.join(os.path.dirname(C.DATA_YAML), "val.txt")
    with open(train_txt, "w") as f:
        f.write("\n".join(train_list))
    with open(val_txt, "w") as f:
        f.write("\n".join(val_list))

    with open(C.DATA_YAML, "w") as f:
        f.write(f"train: {train_txt}\n")
        f.write(f"val: {val_txt}\n")
        f.write("names:\n")
        for i, n in C.NAMES.items():
            f.write(f"  {i}: {n}\n")

    print("=== 데이터셋 구성 ===")
    print(f"blade30 타일 (train)     : {len(b30_train)}")
    print(f"NordTank 라벨 (train)    : {len(nt_labeled)}  (가용 {len(labeled)})")
    print(f"NordTank 배경 (train)    : {len(nt_bg)}  (가용 {len(background)})")
    print(f"----------------------------------------")
    print(f"train 총계               : {len(train_list)}")
    print(f"val (blade30 타일만)     : {len(val_list)}")
    print(f"\nyaml: {C.DATA_YAML}")


if __name__ == "__main__":
    main()
