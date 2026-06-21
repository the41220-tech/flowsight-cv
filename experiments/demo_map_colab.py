"""REAL-footage map-display PoC — runs on Google Colab (network + GPU).

Pipeline:  real VisDrone frame  ->  RT-DETRv2 detection  ->  ground homography
->  people on a 2D map (BEV) + density.  Same frame_to_map() core that the
local render check uses; here the foot points come from a REAL detector on a
REAL downloaded frame.

Colab:
    !pip -q install huggingface_hub datasets "transformers>=4.46" timm accelerate
    !git clone <repo> /content/flowsight   # or set path
    %cd /content/flowsight
    !python experiments/demo_map_colab.py
"""
from __future__ import annotations
import os
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def get_visdrone_frame(idx=0):
    """Stream one real VisDrone frame (no full 1.9GB download)."""
    from datasets import load_dataset
    ds = load_dataset("Voxel51/VisDrone2019-DET", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i == idx:
            return ex["image"].convert("RGB")
    raise RuntimeError("frame index out of range")


def main(idx=0, score_thr=0.25):
    from PIL import Image
    from flowsight.perception.detect import HeadPersonDetector
    from flowsight.pipeline.demo_map import frame_to_map

    img = get_visdrone_frame(idx)
    W, H = img.size
    print(f"real VisDrone frame: {W}x{H}")

    # Aerial people are small -> low threshold. Weak recall here is EXACTLY H3
    # (fine-tune RT-DETR/RF-DETR on DroneCrowd/VisDrone). For dense scenes also
    # try sliced/tiled inference (SAHI) or swap to a crowd-counting head.
    det = HeadPersonDetector(score_thr=score_thr)
    foot = det.foot_points(np.array(img))
    print(f"detected people (foot points): {len(foot)}")

    # --- ground calibration ---------------------------------------------------
    # No survey points for a random web frame, so approximate the visible ground
    # as a trapezoid -> metric rectangle. REPLACE img_quad/map_quad with 4
    # surveyed correspondences for metric-accurate people/m^2 (engine supports
    # it; for non-planar terrain use metric-depth/ray-cast per H1).
    img_quad = np.float32([[0.15 * W, 0.45 * H], [0.85 * W, 0.45 * H], [W, H], [0, H]])
    map_quad = np.float32([[0, 40], [30, 40], [30, 0], [0, 0]])     # assume ~30m x 40m
    bounds = (0, 0, 30, 40)

    res = frame_to_map(np.array(img), foot, img_quad, map_quad, bounds,
                       cell=0.5, sigma_m=1.0,
                       out_path=os.path.join(OUT, "realmap_visdrone.png"),
                       title="VisDrone real frame -> 2D map (RT-DETRv2)")
    print(f"on map: {res['n_on_map']} | BEV peak: {res['max_density']:.1f}/m^2")
    print("saved -> experiments/figures/realmap_visdrone.png")
    return res


if __name__ == "__main__":
    main()
