# Stitches SHARP runs on overlapping tiles of one big room image into one
# splat in the photo camera's frame.
#
# SHARP assumes it looks at the centre of each tile; the real optical centre
# is elsewhere, so each splat is shifted by depth × (tile centre − optical
# centre) / focal. SHARP also guesses scale per run, so each tile's depth is
# matched to a reference room (median ratio where both have surface).
# Overlaps are split down the middle: each pixel belongs to the nearest tile.
#
#   python stitch_tiles.py tiles_in/tiles.json tiles_out reference.ply out.ply

import json, sys
import numpy as np
from scipy.ndimage import median_filter, distance_transform_edt
from plyfile import PlyData, PlyElement

meta = json.load(open(sys.argv[1]))
tiles_out, ref_path, out_path = sys.argv[2], sys.argv[3], sys.argv[4]
BAND = float(sys.argv[5]) if len(sys.argv) > 5 else 0   # photo px; 0 = match everywhere
FIELD = int(sys.argv[6]) if len(sys.argv) > 6 else 0    # >0: per-pixel correction, smoothing size in cells
f, cx, cy = meta["f"], meta["cx"], meta["cy"]
ww, wh = meta["work"]

# reference depth map in work pixels (coarse)
S = 8
ref = PlyData.read(ref_path)["vertex"]
R = np.stack([ref["x"], ref["y"], ref["z"]], 1)
ru = (f * R[:, 0] / R[:, 2] + cx) / S
rv = (f * R[:, 1] / R[:, 2] + cy) / S
gw, gh = ww // S + 1, wh // S + 1
ok = (ru >= 0) & (ru < gw) & (rv >= 0) & (rv < gh)
refd = np.full((gh, gw), np.inf)
np.minimum.at(refd, (rv[ok].astype(int), ru[ok].astype(int)), R[ok, 2])

centres = np.array([[t["x0"] + t["w"] / 2, t["y0"] + t["h"] / 2] for t in meta["tiles"]])
parts = []
for i, t in enumerate(meta["tiles"]):
    v = PlyData.read(f"{tiles_out}/{t['name']}.ply")["vertex"].data.copy()
    x, y, z = (np.asarray(v[k], np.float64) for k in ("x", "y", "z"))
    # tile camera -> photo camera (same focal, optical centre moved)
    ox = t["x0"] + t["w"] / 2 - cx
    oy = t["y0"] + t["h"] / 2 - cy
    x = x + z * ox / f
    y = y + z * oy / f
    u = f * x / z + cx
    w = f * y / z + cy
    # keep only this tile's share of the overlaps
    d = ((u[:, None] - centres[None, :, 0]) ** 2 + (w[:, None] - centres[None, :, 1]) ** 2)
    mine = (d.argmin(1) == i) & (u >= t["x0"]) & (u < t["x0"] + t["w"]) & (w >= t["y0"]) & (w < t["y0"] + t["h"])
    # depth scale against the reference
    iu = np.clip((u / S).astype(int), 0, gw - 1)
    iv = np.clip((w / S).astype(int), 0, gh - 1)
    rd = refd[iv, iu]
    good = mine & np.isfinite(rd)
    # Match where the tiles meet the room: a band just inside the photo's
    # edge (the tiles are only used outside it). A plain scale is stable; the
    # point slides along its own ray, so it stays where the image shows it.
    if BAND:
        ph_x, ph_y = (u - meta["dxw"]) / meta["sw"], (w - meta["dyw"]) / meta["sw"]
        inside = (ph_x > 0) & (ph_x < 3840) & (ph_y > 0) & (ph_y < 2731)
        edge = np.minimum.reduce([ph_x, 3840 - ph_x, ph_y, 2731 - ph_y])
        band = inside & (edge < BAND)
        if (good & band).sum() > 500: good = good & band
    k0 = np.median(rd[good] / z[good]) if good.sum() > 500 else 1.0
    if FIELD:
        # A smooth per-pixel correction: the reference's depth over this
        # tile's depth, median-smoothed, and filled outward from where the
        # reference has surface. The tile then follows the reference's shape
        # (one consistent room) while keeping its own detail.
        sel = mine & np.isfinite(rd)
        tz = np.full((gh, gw), np.inf); np.minimum.at(tz, (iv[sel], iu[sel]), z[sel])
        ratio = np.where(np.isfinite(tz) & np.isfinite(refd), refd / tz, np.nan)
        ok_r = np.isfinite(ratio) & (ratio > k0 * 0.5) & (ratio < k0 * 2)
        ratio = np.where(ok_r, ratio, k0)
        ratio = median_filter(ratio, size=FIELD)
        hole = ~ok_r
        _, (yy, xx) = distance_transform_edt(hole, return_indices=True)
        ratio = np.where(hole, ratio[yy, xx], ratio)
        ratio = median_filter(ratio, size=FIELD)
        k_ = ratio[iv, iu]
    else:
        k_ = k0
    z2 = z * k_
    print(t["name"], "kept", int(mine.sum()), "scale", round(float(k0), 4), "on", int(good.sum()))
    v["x"], v["y"], v["z"] = x * k_, y * k_, z2
    for k in ("scale_0", "scale_1", "scale_2"):
        v[k] = v[k] + np.log(k_)
    parts.append(v[mine])

allv = np.concatenate(parts)
PlyData([PlyElement.describe(allv, "vertex")]).write(out_path)
print("wrote", out_path, len(allv), "splats")
