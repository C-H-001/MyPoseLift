"""Inference-only compatibility shim for MMPose on Windows.

MMPose imports xtcocotools while registering datasets. The inference path only
needs the COCO API, which is provided here by the compatible pycocotools wheel.
"""
