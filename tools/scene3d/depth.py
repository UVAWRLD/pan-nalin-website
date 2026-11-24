# Metric depth + field of view for an image, with Apple's Depth Pro.
#
#   python tools/scene3d/depth.py <image> <out.npz>
#
# Writes depth (metres, float32, image-sized) and fov_x_deg (horizontal field
# of view Depth Pro estimated, degrees).
import sys
import numpy as np
import torch
from PIL import Image
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

src, dst = sys.argv[1], sys.argv[2]
device = "mps" if torch.backends.mps.is_available() else "cpu"
image = Image.open(src).convert("RGB")

processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device).eval()

inputs = processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
post = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])[0]
depth = post["predicted_depth"].float().cpu().numpy()
fov = float(post["field_of_view"])
focal = float(post["focal_length"])
print(f"{src}: depth {depth.min():.3f}..{depth.max():.3f} m, fov {fov:.2f} deg, focal {focal:.1f} px")
np.savez_compressed(dst, depth=depth.astype(np.float32), fov_x_deg=fov, focal_px=focal)
