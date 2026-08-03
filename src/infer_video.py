"""推理演示: 视频/图像 -> RTMPose 2D -> TCN 3D 可视化。

用法: python src/infer_video.py --source video.mp4 --ckpt outputs/ckpt/best.pth
输出: outputs/demo/ 下 3D 骨架图序列 + 合成视频
"""
import sys, argparse
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import CKPT_DIR, RECEPTIVE_FIELD, OUTPUT_DIR, VAL_DIR
from src.model.tcn import TemporalConvNet
from src.data.normalize import center_at_root, normalize_scale, compute_torso_length
from src.visualize import plot_skeleton_3d

# RTMPose 检测 (可选依赖 mmpose; 失败时回退提示)
try:
    from src.detector import RTMPoseDetector
    HAS_DETECTOR = True
except Exception:
    HAS_DETECTOR = False


def build_lifter(ckpt_path, device="cuda"):
    model = TemporalConvNet(34, 17, RECEPTIVE_FIELD, causal=True,
                            num_layers=5, channels=1024).to(device)
    ck = torch_load(ckpt_path, device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def torch_load(path, device):
    import torch
    return torch.load(path, map_location=device)


def normalize_2d_sequence(seq2d):
    """seq2d: (T,17,2) 像素 -> (T,34) 归一化。缺失填 root, torso 缩放。"""
    from src.data.dataset import _fill_missing_with_root
    p2d, _ = _fill_missing_with_root(np.asarray(seq2d, dtype=np.float64))
    centered, _ = center_at_root(p2d)
    raw = np.linalg.norm(centered[:, 5, :] - centered[:, 11, :], axis=-1)
    safe = np.where(raw < 10, np.median(raw), raw)
    safe = np.where(safe < 1e-6, 1.0, safe)
    return (centered / safe[:, None, None]).reshape(len(seq2d), 34), safe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="视频或图像路径")
    parser.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "best.pth"))
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR / "demo"))
    parser.add_argument("--fps", type=float, default=10.0, help="输入视频 fps (T3WB 约 10fps)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    lifter = build_lifter(args.ckpt, device)
    print(f"3D lifting 模型加载: {args.ckpt}")

    # ---- 1. 获取 2D 序列 ----
    if HAS_DETECTOR:
        det = RTMPoseDetector("weights/rtmpose-l_coco_384x288.pth", device=device,
                              cfg_path="/tmp/mmpose_repo/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-384x288.py")
    else:
        det = None
        print("警告: mmpose 不可用, 无法自动检测 2D")

    src = Path(args.source)
    if src.suffix.lower() in (".jpg", ".png"):
        img = cv2.imread(str(src))
        seq2d = []
        if det is not None:
            kpt, conf = det.detect(img, bbox=[0, 0, img.shape[1] - 1, img.shape[0] - 1])
            seq2d.append(kpt)
        # 单帧: 复制作为历史帧
        if len(seq2d) == 1:
            seq2d = [seq2d[0]] * RECEPTIVE_FIELD
        seq2d = np.array(seq2d, dtype=np.float32)
        frame_list = [img]
    else:
        cap = cv2.VideoCapture(str(src))
        frames, seq2d = [], []
        bbox = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if det is not None:
                kpt, conf = det.detect(frame, bbox=bbox)
                if kpt is not None:
                    seq2d.append(kpt)
                    frames.append(frame)
            if len(seq2d) >= 500:
                break
        cap.release()
        seq2d = np.array(seq2d, dtype=np.float32)
        frame_list = frames

    print(f"2D 帧数: {len(seq2d)}")
    if len(seq2d) < RECEPTIVE_FIELD:
        print(f"帧数不足 {RECEPTIVE_FIELD}, 复制首帧补齐")
        pad = RECEPTIVE_FIELD - len(seq2d)
        seq2d = np.concatenate([np.repeat(seq2d[:1], pad, axis=0), seq2d], axis=0)

    # ---- 2. TCN 推理 (滑动窗口, 因果) ----
    import torch
    lifter.eval()
    results_3d = []   # 每帧 (17,3) 归一化
    scales = []
    with torch.no_grad():
        for t in range(len(seq2d)):
            w0 = max(0, t - RECEPTIVE_FIELD + 1)
            win = seq2d[w0:t + 1]
            if len(win) < RECEPTIVE_FIELD:
                win = np.concatenate([np.repeat(win[:1], RECEPTIVE_FIELD - len(win), axis=0), win], axis=0)
            x_norm, scale = normalize_2d_sequence(win)
            x = torch.from_numpy(x_norm).float().unsqueeze(0).to(device)
            pred = lifter(x).cpu().numpy()[0]  # (17,3) 归一化
            results_3d.append(pred)
            scales.append(scale[-1])
    results_3d = np.array(results_3d)
    scales = np.array(scales)
    print(f"3D 推理完成: {len(results_3d)} 帧")

    # ---- 3. 反归一化 (归一化单位 -> 原图 torso 尺度) ----
    # 注意: 归一化尺度是"像素 torso 长度", 3D 输出同样除以此尺度,
    # 反归一化乘以 scale 得到"像素单位 3D", 再按图像尺度换算
    results_pix = results_3d * scales[:, None, None]
    print("3D 输出范围 (像素单位):", results_pix.min(), results_pix.max())

    # ---- 4. 可视化: 每 10 帧保存一张 3D 骨架图 ----
    for i in range(0, len(results_pix), max(1, len(results_pix) // 10)):
        out_png = out_dir / f"pose3d_{i:04d}.png"
        plot_skeleton_3d(results_pix[i], out_png, title=f"frame {i}")
    print(f"可视化已保存: {out_dir}/*.png")


if __name__ == "__main__":
    main()
