"""Person/head detector — RT-DETRv2 (Apache-2.0) via HF transformers.

Colab/GPU. torch/transformers are imported lazily inside __init__ so that
`import flowsight.perception.detect` works on a CPU-only box without them.

NOTE (feeds H3): a COCO-pretrained detector is weak on tiny aerial heads;
fine-tune on DroneCrowd / VisDrone for dense top-down crowds.
"""
from __future__ import annotations
import numpy as np

DEFAULT_MODEL = "PekingU/rtdetr_v2_r50vd"   # Apache-2.0; RF-DETR is the accuracy-first alt


class HeadPersonDetector:
    def __init__(self, model_id=DEFAULT_MODEL, device=None, score_thr=0.5, person_class=0):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.score_thr = score_thr
        self.person_class = person_class

    def detect(self, image):
        from PIL import Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        inputs = self.proc(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        sizes = self.torch.tensor([[image.height, image.width]]).to(self.device)
        r = self.proc.post_process_object_detection(out, target_sizes=sizes, threshold=self.score_thr)[0]
        return (r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), r["labels"].cpu().numpy())

    def foot_points(self, image):
        """bbox bottom-center = ground contact point (for homography/raycast)."""
        b, s, l = self.detect(image)
        b = b[l == self.person_class]
        return np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]]) if len(b) else np.zeros((0, 2))

    def head_points(self, image):
        """bbox top-center = head point (better for dense top-down occlusion)."""
        b, s, l = self.detect(image)
        b = b[l == self.person_class]
        return np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 1]]) if len(b) else np.zeros((0, 2))
