"""端到端训练: 图像 -> 2D热图 + 3D (多任务)
用法: python src/train_e2e.py --epochs 30 --subset 10000
"""
import sys, argparse
sys.path.insert(0, "/home/user/ch/MyPoseLift")
import numpy as np
import torch
import torch.nn as nn
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from src.model.e2e import E2EPoseNet
from configs.config import OUTPUT_DIR

IMG_H, IMG_W = 288, 384   # 输入尺寸
HM_H, HM_W = 9, 12        # 热图尺寸 (stride 32)
STRIDE = 32


def make_heatmap(part, img_h=IMG_H, img_w=IMG_W, hm_h=HM_H, hm_w=HM_W, sigma=1.5):
    """part: (17,2) 像素 -> (17,hm_h,hm_w) 高斯热图"""
    hm = np.zeros((17, hm_h, hm_w), dtype=np.float32)
    for j in range(17):
        x = part[j, 0] / STRIDE
        y = part[j, 1] / STRIDE
        if x < 0 or y < 0 or x >= hm_w or y >= hm_h:
            continue
        xx, yy = np.meshgrid(np.arange(hm_w), np.arange(hm_h))
        hm[j] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return hm


class E2EDataset(Dataset):
    def __init__(self, npz_path, subset=None, train=True):
        d = np.load(npz_path, allow_pickle=True)
        self.imgs = d["img_paths"]
        self.parts = d["parts"]
        self.s3ds = d["s3ds"]
        self.train = train
        if subset:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(self.imgs), subset, replace=False)
            self.imgs, self.parts, self.s3ds = self.imgs[idx], self.parts[idx], self.s3ds[idx]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = cv2.imread(self.imgs[i])
        img = cv2.resize(img, (IMG_W, IMG_H))
        part = self.parts[i].copy()
        # 尺度: 像素与 resize 一致 (1000x1002 -> 384x288)
        sx, sy = IMG_W / 1000.0, IMG_H / 1002.0
        part[:, 0] *= sx
        part[:, 1] *= sy
        if self.train and np.random.rand() > 0.5:
            img = img[:, ::-1].copy()
            part[:, 0] = IMG_W - part[:, 0]
            # H36M 17 对称交换
            flip = [(1,4),(2,5),(3,6),(11,14),(12,15),(13,16)]
            for a, b in flip:
                part[[a, b]] = part[[b, a]]
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.transpose(2, 0, 1)  # (3,H,W)
        hm = make_heatmap(part)
        s3d = self.s3ds[i].astype(np.float32)
        return (torch.from_numpy(img).float(),
                torch.from_numpy(hm).float(),
                torch.from_numpy(s3d).float())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--subset", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lam2d", type=float, default=0.1, help="2D loss 权重")
    args = parser.parse_args()

    device = "cuda"
    train_ds = E2EDataset("/home/user/ch/MyPoseLift/data/cache/e2e_h36m.npz",
                          subset=args.subset, train=True)
    # 留 10% 做验证
    n_val = int(len(train_ds) * 0.1)
    val_ds = torch.utils.data.Subset(train_ds, range(n_val))
    train_ds_sub = torch.utils.data.Subset(train_ds, range(n_val, len(train_ds)))
    train_loader = DataLoader(train_ds_sub, batch_size=args.batch, shuffle=True,
                              num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    model = E2EPoseNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    mse = nn.MSELoss()

    print(f"训练样本: {len(train_ds_sub)}, 验证: {len(val_ds)}, epochs: {args.epochs}")
    for epoch in range(args.epochs):
        model.train()
        tot3d, tot2d, nb = 0, 0, 0
        for img, hm, s3d in train_loader:
            img, hm, s3d = img.to(device), hm.to(device), s3d.to(device)
            opt.zero_grad()
            p3d, hm_pred = model(img)
            loss3d = torch.norm(p3d - s3d, dim=-1).mean()
            loss2d = mse(hm_pred, hm)
            loss = loss3d + args.lam2d * loss2d
            loss.backward()
            opt.step()
            tot3d += loss3d.item(); tot2d += loss2d.item(); nb += 1
        sched.step()

        # 验证
        model.eval()
        v3d, v2d = 0, 0
        with torch.no_grad():
            for img, hm, s3d in val_loader:
                img, hm, s3d = img.to(device), hm.to(device), s3d.to(device)
                p3d, hm_pred = model(img)
                v3d += torch.norm(p3d - s3d, dim=-1).mean().item()
                v2d += mse(hm_pred, hm).item()
        v3d /= len(val_loader); v2d /= len(val_loader)
        print(f"Epoch {epoch:3d} | train3d {tot3d/nb*1000:.1f}mm 2d {tot2d/nb:.4f} | "
              f"val3d {v3d*1000:.1f}mm 2d {v2d:.4f}", flush=True)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val3d": v3d}, f"/home/user/ch/MyPoseLift/outputs/ckpt/e2e_{epoch:03d}.pth")

    print("端到端训练完成")


if __name__ == "__main__":
    main()
