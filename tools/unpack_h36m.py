"""解压 h36m.zip 到 /mnt/disk2/ch/H36M/images/ (约 3.6G, S1/S5/S7)"""
import zipfile
from pathlib import Path

ZIP = Path("/mnt/disk2/ch/H36M/h36m.zip")
OUT = Path("/mnt/disk2/ch/H36M/images")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        members = [m for m in z.namelist() if m.startswith("original/") and m.endswith(".jpg")]
        print(f"共 {len(members)} 张图像", flush=True)
        for i, m in enumerate(members):
            # original/S1/Images/Directions 1.54138969/frame_0000.jpg
            rel = m.replace("original/", "")
            target = OUT / rel
            if target.exists() and target.stat().st_size > 1000:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(members)}", flush=True)
    print("解压完成", flush=True)


if __name__ == "__main__":
    main()
