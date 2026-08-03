"""3DPW test 缓存 (与 train 相同处理)"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.joint_mapping import build_coco17_supervision_mask
from tools.prepare_pw3d import load_sequence, seq_to_coco
from configs.config import CACHE_DIR, PW3D_DIR

SEQ_ROOT = PW3D_DIR / "sequenceFiles" / "sequenceFiles" / "test"
OUT_PATH = CACHE_DIR / "pw3d_test.npz"


def build():
    out = {}
    for pkl_path in sorted(Path(SEQ_ROOT).glob("*.pkl")):
        seq_name = pkl_path.stem
        try:
            data = load_sequence(pkl_path)
        except Exception as e:
            print(f"  跳过 {seq_name}: {e}")
            continue
        p2d, c3d, fids, valid = seq_to_coco(data)
        out[seq_name] = {seq_name: {"cam0": {
            "pose2d_coco17": p2d,
            "cam3d_coco17": c3d,
            "world3d_coco17": c3d,
            "frame_id": fids,
        }}}
        print(f"  {seq_name}: T={len(fids)}")
    np.savez(OUT_PATH, data=out, supervision_mask=build_coco17_supervision_mask())
    print(f"3DPW test 缓存: {OUT_PATH}, 序列数: {len(out)}")


if __name__ == "__main__":
    build()
