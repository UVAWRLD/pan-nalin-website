# Add the chest hidden behind his arm: splats from SHARP(photo with the arm
# painted out), only inside the arm mask and only behind the current boy.
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.ndimage import minimum_filter
F, CX, CY, OX, OY, S = 5268.6147, 1885.5, 1285.0, 69, 161, 4
W, H = 3840 // S, 2731 // S
def load(p):
    v = PlyData.read(p)["vertex"]; return v, np.stack([v["x"], v["y"], v["z"]], 1)
def pix(P): return (F*P[:,0]/P[:,2]+CX+OX)/S, (F*P[:,1]/P[:,2]+CY+OY)/S
boy_v, B = load("boy_splat.ply")
new_v, N = load("sharp_out_noarm/noarm.ply")
orig_v, O = load("sharp_out_boy/photo.ply")
occ = np.asarray(Image.open("occluder_mask.png").convert("L").resize((W, H), Image.NEAREST)) > 0
# depth scale: compare SHARP(no arm) with SHARP(original) outside the mask
def dmap(P):
    u, v = pix(P); ok = (u>=0)&(u<W)&(v>=0)&(v<H)
    d = np.full((H, W), np.inf); np.minimum.at(d, (v[ok].astype(int), u[ok].astype(int)), P[ok, 2]); return d
dn, do = dmap(N), dmap(O)
both = np.isfinite(dn) & np.isfinite(do) & ~occ & (do < 2.6)
r = np.median(do[both] / dn[both]); N = N * r
print("scale", round(r, 4))
near = minimum_filter(dmap(B), 3)        # the current boy's nearest surface
u, v = pix(N); iu = np.clip(u.astype(int), 0, W-1); iv = np.clip(v.astype(int), 0, H-1)
inside = (u>=0)&(u<W)&(v>=0)&(v<H) & occ[iv, iu]
alpha = 1 / (1 + np.exp(-np.asarray(new_v["opacity"])))
# Only skin at chest depth: inside the film loop the painted fill is wall,
# which must stay out of the boy.
C0 = 0.28209479177387814
rgb = np.clip(np.stack([new_v["f_dc_0"], new_v["f_dc_1"], new_v["f_dc_2"]], 1) * C0 + 0.5, 0, 1)
mx, mn = rgb.max(1), rgb.min(1)
skin = ((mx - mn) / np.maximum(mx, 1e-3) > 0.3) & (rgb[:, 0] >= rgb[:, 1]) & (rgb[:, 1] >= rgb[:, 2]) & (mx > 0.12)
torso = (N[:, 2] > 1.85) & (N[:, 2] < 2.3)
keep = inside & (alpha > 0.2) & (N[:, 2] > near[iv, iu] + 0.06) & torso & skin
print("hidden chest splats", keep.sum())
d = new_v.data[keep].copy()
for k, i in (("x", 0), ("y", 1), ("z", 2)): d[k] = N[keep, i]
for k in ("scale_0", "scale_1", "scale_2"): d[k] += np.log(r)
names = boy_v.data.dtype.names
merged = np.concatenate([boy_v.data[list(names)], d[list(names)]])
PlyData([PlyElement.describe(merged, "vertex")]).write("boy_with_chest.ply")
