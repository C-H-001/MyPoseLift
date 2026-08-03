"""T3WB 世界->相机 一致性回归测试 (T 单位米的关键验证)"""
import numpy as np
from src.data.t3wb import load_t3wb_action, get_camera_params
from src.data.camera import world_to_camera


def test_world_to_camera_consistency():
    """Xc = R @ Xw + T_mm 应与 T3WB 提供的 camera_3d 一致 (误差 < 1mm)"""
    for subj, act, ck in [("S5", "Directions 1", "60457274"),
                          ("S1", "Walking", "55011271"),
                          ("S7", "Directions", "54138969")]:
        data = load_t3wb_action(subj, act)
        K, R, T, D = get_camera_params(subj, ck)
        g3d = np.asarray(data["global_3d"][:50], dtype=np.float32)   # (50,133,3) 世界系
        c3d = np.asarray(data[ck]["camera_3d"][:50], dtype=np.float32)  # (50,133,3) 相机系
        conv = world_to_camera(g3d, R, T)
        err = float(np.abs(conv - c3d).mean())
        assert err < 1.0, f"{subj}/{act}/{ck}: 世界->相机 误差 {err:.2f}mm > 1mm"


def test_t_unit_converted():
    """T 已转换为 mm 单位"""
    K, R, T, D = get_camera_params("S1", "60457274")
    assert abs(T[0, 0, 2] - 4.406149 * 1000) < 1.0  # 4.406m -> 4406mm
