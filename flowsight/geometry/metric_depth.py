"""Monocular metric depth -> 3D ground points (the depth path for H1).

Two strategies:
  1) RELATIVE depth (Depth-Anything-V2-*-hf, well-supported in transformers)
     scaled to METERS via a few known anchor distances (robust, license-clean).
  2) NATIVE metric checkpoint (Depth-Anything-V2-Metric-VKITTI-Large, Apache-2.0)
     when set up via the official depth_anything_v2 package.

Lazy heavy imports -> CPU-importable. backproject() reuses PinholeCamera.
"""
from __future__ import annotations
import numpy as np

REL_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"     # relative, transformers-native
METRIC_MODEL = "depth-anything/Depth-Anything-V2-Metric-VKITTI-Large"  # native metric (official repo)


class MetricDepth:
    def __init__(self, model_id=REL_MODEL, device=None):
        import torch
        from transformers import pipeline
        self.torch = torch
        dev = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("depth-estimation", model=model_id, device=dev)

    def predict(self, image):
        """Returns the model's depth map as a float array (relative unless a
        metric checkpoint is used)."""
        from PIL import Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        out = self.pipe(image)
        d = out.get("predicted_depth", out.get("depth"))
        return np.asarray(d, dtype=float)

    @staticmethod
    def scale_to_metric(depth_map, pixel_anchors, metric_Z):
        """Fit an affine map model-output -> metres on known anchors:
        Z_metric ~ a * model + b (least squares). Use >=2 ground anchors whose
        true camera-Z you know (e.g. from homography anchor distances)."""
        pixel_anchors = np.asarray(pixel_anchors, int)
        vals = depth_map[pixel_anchors[:, 1], pixel_anchors[:, 0]]
        A = np.column_stack([vals, np.ones_like(vals)])
        a, b = np.linalg.lstsq(A, np.asarray(metric_Z, float), rcond=None)[0]
        return a * depth_map + b

    @staticmethod
    def backproject(camera, foot_pixels, metric_depth_map):
        """foot pixels (N,2) + metric depth map -> world (N,3) via the camera."""
        foot = np.asarray(foot_pixels, float)
        z = metric_depth_map[np.clip(foot[:, 1].astype(int), 0, metric_depth_map.shape[0] - 1),
                             np.clip(foot[:, 0].astype(int), 0, metric_depth_map.shape[1] - 1)]
        return camera.backproject_depth(foot, z)
