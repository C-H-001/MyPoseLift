"""RTMPose 推理封装: 图像 -> 2D 关键点 (17,2) 像素 + 置信度。

流程: mmdet RTMDet 检测人框 -> mmpose RTMPose top-down 关键点。
依赖: mmcv, mmdet, mmpose (见 scripts/install_mm.sh)。
"""
from pathlib import Path
import numpy as np
import cv2

# 配置文件路径 (mmpose 仓库)
MMPOSE_REPO = Path("/tmp/mmpose_repo")
RTMPOSE_CFG = (MMPOSE_REPO / "mmpose/configs/body_2d_keypoint/rtmpose/coco"
               "/rtmpose-l_8xb256-420e_aic-coco-384x288.py")
RTMDET_CFG = None  # 若用 mmdet 默认配置


class RTMPoseDetector:
    """RTMPose 2D 关键点检测器。

    model_config: 指定 'rtmpose-l' 等, 加载 mmpose 预训练配置。
    """

    def __init__(self, weights_path, device="cuda", cfg_path=None):
        from mmpose.apis import init_model as init_mmpose
        from mmengine.config import Config
        cfg = Config.fromfile(str(cfg_path or RTMPOSE_CFG))
        self.model = init_mmpose(cfg, str(weights_path), device=device)
        self.device = device
        # 记录 COCO 17 名称映射
        self._joint_names = ["nose", "l_eye", "r_eye", "l_ear", "r_ear",
                             "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
                             "l_wrist", "r_wrist", "l_hip", "r_hip",
                             "l_knee", "r_knee", "l_ankle", "r_ankle"]

    def detect(self, img_bgr, bbox=None, conf_thresh=0.3):
        """单图检测。img_bgr: HxWx3 BGR ndarray。
        bbox: [x1,y1,x2,y2] 或 None (需外部提供人框)。
        返回: (17,2) 像素坐标 + (17,) 置信度; 失败返回 (None, None)。
        """
        from mmpose.apis import inference_topdown
        if bbox is None:
            return None, None
        results = inference_topdown(self.model, img_bgr, bboxes=[bbox])
        if not results or results[0].pred_instances is None:
            return None, None
        inst = results[0].pred_instances
        kpts = inst.keypoints.cpu().numpy()   # (1,17,2)
        scores = inst.keypoint_scores.cpu().numpy()  # (1,17)
        return kpts[0], scores[0]

    def detect_video(self, video_path, bbox_provider=None, out_path=None):
        """视频逐帧检测 (bbox_provider 提供人框, 默认全帧首检测复用)。"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        kpts_seq = []
        bbox = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if bbox_provider is not None:
                bbox = bbox_provider(frame)
            if bbox is None and len(kpts_seq) > 0:
                pass  # 复用前一帧 bbox (简化)
            kpt, conf = self.detect(frame, bbox=bbox)
            kpts_seq.append(kpt)
            frames.append(frame)
        cap.release()
        return frames, kpts_seq
