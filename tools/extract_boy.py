# Keep the boy's splats from SHARP(original photo): inside his cutout and in
# front of the empty room (SHARP of the boy-free image) at the same pixel.
import sys, numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.ndimage import binary_dilation, binary_erosion, minimum_filter
F, CX, CY, OX, OY = 5268.6147, 1885.5, 1285.0, 69, 161
def load(path):
    v = PlyData.read(path)["vertex"]; return v, np.stack([v["x"], v["y"], v["z"]], 1)
boy_v, B = load("sharp_out_boy/photo.ply")
_, R = load("sharp_out_mj/room_mj.ply")
S = 4  # depth grid at 1/4 photo px
W, H = 3840 // S, 2731 // S
def pix(P):
    return (F * P[:, 0] / P[:, 2] + CX + OX) / S, (F * P[:, 1] / P[:, 2] + CY + OY) / S
ru, rv = pix(R); ok = (ru >= 0) & (ru < W) & (rv >= 0) & (rv < H)
room = np.full((H, W), 99.0)
np.minimum.at(room, (rv[ok].astype(int), ru[ok].astype(int)), R[ok, 2])
room = minimum_filter(room, 3)
m = np.asarray(Image.open("boy_mask.png").convert("L").resize((W, H), Image.BILINEAR)) > 110
# The cutout model also grabbed the trunk on the left. Don't cut it with a
# straight line (that sliced his knee): left of x=1032 px, keep only what is
# clearly in front of the room, which his knee is and the trunk is not.
trunk_zone = np.zeros_like(m); trunk_zone[:, : 1032 // S] = True
core = binary_erosion(m, iterations=3)          # well inside his outline
m = binary_dilation(m, iterations=int(sys.argv[1]) if len(sys.argv) > 1 else 1)
bu, bv = pix(B); okb = (bu >= 0) & (bu < W) & (bv >= 0) & (bv < H)
iu = np.clip(bu.astype(int), 0, W - 1); iv = np.clip(bv.astype(int), 0, H - 1)
# Near his outline keep only what is in front of the room (no wall rim);
# well inside it keep everything, including the film on the floor in his lap,
# which sits at floor depth.
front = B[:, 2] < room[iv, iu] * 0.94
# SHARP builds each surface in two layers; behind his hand, hair and knees the
# second layer is a blurry copy further back, which shows as a ghost when the
# camera moves. Keep only splats within a body's thickness of his nearest
# surface at that pixel.
alpha = 1 / (1 + np.exp(-np.asarray(boy_v["opacity"])))
cand = okb & m[iv, iu] & (alpha > 0.3)
near = np.full((H, W), 99.0)
np.minimum.at(near, (iv[cand], iu[cand]), B[cand, 2])
near = minimum_filter(near, 3)
THICK = float(sys.argv[2]) if len(sys.argv) > 2 else 0.18
skin_deep = B[:, 2] < near[iv, iu] + THICK
# Nothing of him is farther than ~2.5 (his back is ~2.3): deeper splats
# are bits of drum and wall the cutout let through.
keep = okb & m[iv, iu] & skin_deep & (B[:, 2] < 2.5) & (front | (core[iv, iu] & ~trunk_zone[iv, iu]))
print("boy splats", keep.sum(), "of", len(B), "depth pct", np.percentile(B[keep, 2], [5, 50, 95]).round(2))
PlyData([PlyElement.describe(boy_v.data[keep], "vertex")]).write("boy_splat.ply")
