"""训练入口: python src/train.py [--epochs 50] [--resume ckpt.pt] [--batch 1024]

每轮: 训练 -> val 加权 MPJPE -> 5 样本可视化 (GT vs Pred) -> checkpoint
数据: T3WB 缓存 (S1,S5,S6,S7 训练; S5 部分做开发验证; S8 由评测脚本评估)
"""
import argparse
import time
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (EPOCHS, BATCH_SIZE, LR, RECEPTIVE_FIELD, SEED,
                            CKPT_DIR, VAL_DIR, LOG_DIR, CACHE_DIR, NUM_WORKERS)
from src.data.dataset import TemporalPoseDataset
from src.data.joint_mapping import build_coco17_supervision_mask
from src.model.tcn import TemporalConvNet
from src.losses import weighted_mpjpe_loss
from src.visualize import plot_skeleton_3d

TRAIN_SUBJECTS = ["S1", "S5", "S6", "S7"]
VAL_SUBJECTS = ["S5"]  # 开发验证 (正式评测用 S8 test npz)


def build_model(rf=81):
    return TemporalConvNet(num_input_channels=34, num_joints=17,
                           receptive_field=rf, causal=True,
                           num_layers=5, channels=1024)


def evaluate(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            x, y, m = batch[0], batch[1], batch[2]
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x)
            losses.append(weighted_mpjpe_loss(pred, y, m).item())
    return float(np.mean(losses)) if losses else float("inf")


def save_val_samples(model, loader, device, epoch, max_save=5):
    """保存 N 个样本: 3D GT vs Pred 对比图 (归一化空间, torso=1)"""
    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)
            pred = model(x)
            for i in range(min(max_save - saved, x.size(0))):
                p = pred[i].cpu().numpy()   # (17,3) 归一化
                g = y[i].cpu().numpy()
                out = VAL_DIR / f"epoch{epoch:03d}_sample{saved}.png"
                # 对比图: GT / Pred
                from matplotlib import pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                                         subplot_kw={"projection": "3d"})
                for ax, X, ttl in [(axes[0], g, "GT"), (axes[1], p, "Pred")]:
                    for (j, k) in [(5, 7), (7, 9), (6, 8), (8, 10),
                                   (5, 11), (6, 12), (11, 13), (13, 15),
                                   (12, 14), (14, 16), (11, 12)]:
                        if np.isnan(X[j]).any() or np.isnan(X[k]).any():
                            continue
                        ax.plot([X[j, 0], X[k, 0]], [X[j, 1], X[k, 1]],
                                [X[j, 2], X[k, 2]], "o-", lw=2)
                    lim = 2.0
                    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
                    ax.set_title(ttl)
                fig.suptitle(f"epoch {epoch} sample {i} (normalized units)")
                fig.savefig(out, dpi=100)
                plt.close(fig)
                saved += 1
            if saved >= max_save:
                break
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--datasets", type=str, default="t3wb",
                        help="逗号分隔: t3wb,pw3d")
    parser.add_argument("--rf", type=int, default=RECEPTIVE_FIELD,
                        help="感受野 (窗口帧数)")
    parser.add_argument("--pw3d-weight", type=float, default=8.0,
                        help="3DPW 采样权重 (平衡小数据集), 默认 8 (约 1:2 比例)")
    parser.add_argument("--augment", action="store_true",
                        help="启用时序数据增强 (3D->投影2D+噪声+dropout)")
    args = parser.parse_args()
    rf = args.rf

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 数据 (支持多数据集混合 + 平衡采样)
    from torch.utils.data import ConcatDataset, WeightedRandomSampler
    datasets = args.datasets.split(",")
    train_dss, val_dss, ds_weights = [], [], []
    for ds_name in datasets:
        if ds_name == "t3wb":
            d = TemporalPoseDataset(CACHE_DIR / "t3wb_train.npz",
                                    subjects=TRAIN_SUBJECTS, rf=rf)
            train_dss.append(d); ds_weights.append(1.0)
            val_dss.append(TemporalPoseDataset(CACHE_DIR / "t3wb_train.npz",
                                               subjects=VAL_SUBJECTS, rf=rf, stride=5))
        elif ds_name == "pw3d":
            d = TemporalPoseDataset(CACHE_DIR / "pw3d_train.npz", rf=rf)
            train_dss.append(d); ds_weights.append(args.pw3d_weight)
        elif ds_name == "h36m":
            # H36M 官方 10fps 全量: 训练 S1,S5,S6,S7,S8, 验证 S9,S11 (标准协议)
            d = TemporalPoseDataset(CACHE_DIR / "h36m_train.npz", rf=rf)
            train_dss.append(d); ds_weights.append(1.0)
            val_dss.append(TemporalPoseDataset(CACHE_DIR / "h36m_valid.npz",
                                               rf=rf, stride=10))
        elif ds_name == "h36m_h17":
            # H36M 17 点原序 (全 17 点监督, 与 H36M 2D 检测器一致)
            # root_idx=0: H36M 关节 0 (Hip) 作为根 (VideoPose3D 协议)
            if args.augment:
                from src.data.aug_dataset import AugmentedPoseDataset
                from src.augmentation.sequence_augmentation import ResidualBank
                bank = ResidualBank.from_array(
                    np.load(CACHE_DIR / "residual_bank.npy"))
                d = AugmentedPoseDataset(CACHE_DIR / "h36m_train_h17.npz", rf=rf,
                                         residual_bank=bank)
            else:
                d = TemporalPoseDataset(CACHE_DIR / "h36m_train_h17.npz", rf=rf, root_idx=0)
            train_dss.append(d); ds_weights.append(1.0)
            val_dss.append(TemporalPoseDataset(CACHE_DIR / "h36m_valid_h17.npz",
                                               rf=rf, stride=10, root_idx=0))
    train_ds = ConcatDataset(train_dss)
    if len(val_dss) == 0:
        val_ds = train_dss[0]  # 无验证集时退化
    elif len(val_dss) == 1:
        val_ds = val_dss[0]
    else:
        val_ds = ConcatDataset(val_dss)

    # 平衡采样: 每个样本权重 = ds_weight / len(ds) (3DPW 等小数据集上采样)
    sample_weights = []
    for d, w in zip(train_dss, ds_weights):
        sample_weights.extend([w / len(d)] * len(d))
    train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                          replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_model(rf).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    writer = SummaryWriter(LOG_DIR)

    start_epoch = 0
    best = float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        best = ck.get("val", float("inf"))
        print(f"恢复自 {args.resume} (epoch {ck['epoch']}, best val {best:.4f})")

    print(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")
    print(f"每轮 batch: {len(train_loader)}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for batch in train_loader:
            x, y, m = batch[0], batch[1], batch[2]
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = weighted_mpjpe_loss(pred, y, m)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        train_loss = tot / nb

        val_loss = evaluate(model, val_loader, device)
        n_saved = save_val_samples(model, val_loader, device, epoch)
        writer.add_scalar("train/mpjpe_norm", train_loss, epoch)
        writer.add_scalar("val/mpjpe_norm", val_loss, epoch)
        print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| 样本{'{'}{n_saved}{'}'} | {time.time()-t0:.0f}s", flush=True)

        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": epoch, "val": val_loss}, CKPT_DIR / "best.pth")
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": epoch}, CKPT_DIR / f"epoch_{epoch:03d}.pth")

    print(f"训练完成。最佳 val MPJPE(归一化): {best:.4f} -> best.pth")


if __name__ == "__main__":
    main()
