# Alpha matte from an image and a trimap, with ViTMatte, run in overlapping tiles
# so a 4K image keeps its full detail (hair strands).
#
#   python tools/scene3d/matte.py <image> <trimap.png> <alpha_out.png> [tile=1280] [overlap=320]
#
# trimap: 0 = background, 255 = foreground, anything else = unknown.
import sys
import numpy as np
import torch
from PIL import Image
from transformers import VitMatteImageProcessor, VitMatteForImageMatting

src, tri_path, dst = sys.argv[1:4]
TILE = int(sys.argv[4]) if len(sys.argv) > 4 else 1280
OVER = int(sys.argv[5]) if len(sys.argv) > 5 else 320
device = "mps" if torch.backends.mps.is_available() else "cpu"

image = np.asarray(Image.open(src).convert("RGB"))
tri = np.asarray(Image.open(tri_path).convert("L"))
tri = np.where(tri < 20, 0, np.where(tri > 235, 255, 128)).astype(np.uint8)
H, W = tri.shape

name = "hustvl/vitmatte-base-composition-1k"
processor = VitMatteImageProcessor.from_pretrained(name)
model = VitMatteForImageMatting.from_pretrained(name).to(device).eval()

acc = np.zeros((H, W), np.float32)
wsum = np.zeros((H, W), np.float32)
step = TILE - OVER


def starts(n):
    if n <= TILE:
        return [0]
    s = list(range(0, n - TILE, step)) + [n - TILE]
    return sorted(set(s))


for y0 in starts(H):
    for x0 in starts(W):
        y1, x1 = min(H, y0 + TILE), min(W, x0 + TILE)
        t = tri[y0:y1, x0:x1]
        if not (t == 128).any():
            a = (t == 255).astype(np.float32)
        else:
            inp = processor(images=Image.fromarray(image[y0:y1, x0:x1]), trimaps=Image.fromarray(t),
                            return_tensors="pt").to(device)
            with torch.no_grad():
                a = model(**inp).alphas[0, 0, : y1 - y0, : x1 - x0].float().cpu().numpy()
        # weight: high in the middle of a tile, low at its edges
        wy = np.minimum(np.arange(y1 - y0) + 1, np.arange(y1 - y0)[::-1] + 1).astype(np.float32)
        wx = np.minimum(np.arange(x1 - x0) + 1, np.arange(x1 - x0)[::-1] + 1).astype(np.float32)
        w = np.minimum(np.minimum.outer(wy, wx), OVER) + 1e-3
        acc[y0:y1, x0:x1] += a * w
        wsum[y0:y1, x0:x1] += w
        print("tile", x0, y0, flush=True)

alpha = acc / wsum
alpha[tri == 0] = 0
alpha[tri == 255] = 1
Image.fromarray((np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)).save(dst)
print("saved", dst)
