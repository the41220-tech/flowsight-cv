"""Static obstacle / geofence layer + object identification (H4).

- Open-vocabulary detection: Grounding DINO (Apache-2.0) with text prompts like
  "fence. barrier. stage. water. stairs. wall." — catches safety-relevant static
  structures even if absent from COCO.
- (optional) Panoptic seg: Mask2Former for pixel masks of those structures.
- VLM identify: Qwen2.5-VL names/captions an ambiguous crop ("what is this?").

Detected obstacles are projected to the ground map (via the same img->map
function used for people) to auto-build a geofence / impassable layer.
Lazy heavy imports -> CPU-importable.
"""
from __future__ import annotations
import numpy as np

SAFETY_PROMPTS = ["fence", "barrier", "stage", "water", "stairs", "wall",
                  "gate", "vehicle", "tent", "pole"]


class OpenVocabObstacles:
    def __init__(self, model_id="IDEA-Research/grounding-dino-base", device=None, box_thr=0.35, text_thr=0.25):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.box_thr, self.text_thr = box_thr, text_thr

    def detect(self, image, prompts=SAFETY_PROMPTS):
        from PIL import Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        text = ". ".join(prompts) + "."
        inputs = self.proc(images=image, text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            out, inputs.input_ids, box_threshold=self.box_thr, text_threshold=self.text_thr,
            target_sizes=[(image.height, image.width)])[0]
        return (res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), res.get("labels"))

    @staticmethod
    def to_geofence(boxes, img2map_fn):
        """Project obstacle bbox bottom edge to the ground map -> impassable
        segments. img2map_fn: (N,2 pixels)->(N,2 metres)."""
        polys = []
        for x1, y1, x2, y2 in boxes:
            base = np.array([[x1, y2], [x2, y2]])     # ground contact edge
            polys.append(img2map_fn(base))
        return polys


class VLMIdentifier:
    """Qwen2.5-VL (Apache-2.0) — name/caption an ambiguous crop. Used only for
    objects/obstacles, never for person identity (privacy by design)."""
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", device=None):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto")

    def identify(self, image_crop, question="What is this object? Is it a crowd-safety hazard? Answer briefly."):
        from PIL import Image
        if isinstance(image_crop, np.ndarray):
            image_crop = Image.fromarray(image_crop)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image_crop}, {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[image_crop], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=64)
        trimmed = ids[:, inputs.input_ids.shape[1]:]
        return self.proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
