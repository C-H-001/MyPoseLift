"""评测脚本: 加载 best.pth, 在 T3WB S8 test (task1/task2) 上计算 MPJPE。

输出:
- 归一化空间 MPJPE (torso=1)
- 真实 mm MPJPE (乘训练集 median torso 长度)
"""
import sys, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import CKPT_DIR, RECEPTIVE_FIELD, CACHE_DIR, NUM_WORKERS
from src.data.dataset import TemporalPoseDataset
from src.data.joint_mapping import build_coco17_supervision_mask
from src.data.normalize import compute_torso_length
from src.model.tcn import TemporalConvNet
from src.losses import weighted_mpjpe_loss

# 训练集 median torso 长度 (mm) - 用于归一化单位 -> mm 换算
TRAIN_TORSO_MM = 450.0  # 从 check_data 的骨架分析: torso ~437-451mm


def eval_npz(model, npz_path, device, joint_mask, batch=512, verbose=False):
    ds = TemporalPoseDataset(npz_path, subjects=None, rf=RECEPTIVE_FIELD)
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    errs = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            d = torch.norm(pred - y, dim=-1)  # (B,17)
            errs.append(d.cpu().numpy())
    errs = np.concatenate(errs, 0)
    mask = joint_mask.numpy().astype(bool)
    sup = errs[:, mask]
    if verbose:
        names = ["nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder",
                 "l_elbow","r_elbow","l_wrist","r_wrist","l_hip","r_hip",
                 "l_knee","r_knee","l_ankle","r_ankle"]
        for j in range(17):
            if mask[j]:
                print(f"  {names[j]:12s}: {errs[:, j].mean():.4f} (norm)")
    return float(errs[:, sup].mean()), float(np.median(errs[:, sup]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "best.pth"))
    parser.add_argument("--torso-mm", type=float, default=TRAIN_TORSO_MM)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TemporalConvNet(34, 17, RECEPTIVE_FIELD, causal=True,
                            num_layers=5, channels=1024).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"加载模型: {args.ckpt} (epoch {ck['epoch']})")
    print(f"torso 换算: 1 归一化单位 = {args.torso_mm} mm")
    print()

    mask = torch.from_numpy(build_coco17_supervision_mask()).float().to(device)
    for npz_name in ["t3wb_test1.npz", "t3wb_test2.npz"]:
        mean, med = eval_npz(model, CACHE_DIR / npz_name, device, mask, verbose=True)
        print(f"\n[{npz_name}] MPJPE: {mean:.4f} (norm) = {mean * args.torso_mm:.1f} mm | "
              f"median {med:.4f} (norm) = {med * args.torso_mm:.1f} mm")


if __name__ == "__main__":
    main()
