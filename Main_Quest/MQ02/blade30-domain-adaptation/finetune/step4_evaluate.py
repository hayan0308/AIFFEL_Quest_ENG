"""
평가 ④ : 파인튜닝 전(best.pt) vs 후(파인튜닝 결과)를 blade30 val(target 도메인)에서 비교한다.
같은 val 셋으로 mAP를 재서 개선폭을 정직하게 확인한다.
"""
import os, sys
import config as C
from ultralytics import YOLO


def evaluate(weights, tag):
    if not os.path.exists(weights):
        print(f"[skip] {tag}: 가중치 없음 ({weights})")
        return None
    print(f"\n===== {tag} : {weights} =====")
    m = YOLO(weights)
    r = m.val(data=C.DATA_YAML, split="val", project=C.RUNS_DIR, name=f"eval_{tag}")
    return r.box.map, r.box.map50


def main():
    ft = os.path.join(C.RUNS_DIR, "blade30_finetune", "weights", "best.pt")
    before = evaluate(C.BASE_WEIGHTS, "before")
    after  = evaluate(ft, "after")

    print("\n================ 요약 (blade30 val) ================")
    if before:
        print(f"파인튜닝 전  mAP50={before[1]:.3f}  mAP50-95={before[0]:.3f}")
    if after:
        print(f"파인튜닝 후  mAP50={after[1]:.3f}  mAP50-95={after[0]:.3f}")


if __name__ == "__main__":
    main()
