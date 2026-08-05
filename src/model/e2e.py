"""端到端 17 点模型: 图像 -> CSPNeXt -> 2D热图 + 3D回归
多任务: 2D 辅助监督 (part 热图) + 3D 主监督 (root相对米制)
"""
import torch
import torch.nn as nn


class E2EPoseNet(nn.Module):
    def __init__(self, num_joints=17, input_size=(288, 384), pretrained=True):
        super().__init__()
        self.num_joints = num_joints
        self.input_size = input_size  # (W, H)

        # CSPNeXt backbone (mmdet)
        from mmdet.models.backbones import CSPNeXt
        self.backbone = CSPNeXt(
            arch='P5', expand_ratio=0.5, deepen_factor=1., widen_factor=1.,
            out_indices=(4,), channel_attention=True,
            norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        if pretrained:
            import torch.hub as hub
            # 加载 cspnext-l 预训练 (从 mmpose 权重提取 backbone)
            try:
                state = torch.hub.load_state_dict_from_url(
                    'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
                    'cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth',
                    map_location='cpu')
                sd = {k.replace('backbone.', ''): v for k, v in state['state_dict'].items()
                      if k.startswith('backbone.')}
                self.backbone.load_state_dict(sd, strict=False)
                print("backbone 预训练加载 OK")
            except Exception as e:
                print("预训练加载失败:", e)

        # 2D 热图头 (低分辨率 12x9)
        self.head_2d = nn.Conv2d(1024, num_joints, 1)

        # 3D 回归头 (GAP -> MLP -> 17x3)
        self.head_3d = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, num_joints * 3))

    def forward(self, img):
        """img: (B,3,H,W) -> (3D: B,17,3), (2D热图: B,17,h,w)"""
        feat = self.backbone(img)  # [(B,1024,h,w)]
        feat = feat[-1]
        hm = self.head_2d(feat)      # (B,17,h,w)
        p3d = self.head_3d(feat)     # (B,51)
        p3d = p3d.reshape(-1, self.num_joints, 3)
        return p3d, hm
