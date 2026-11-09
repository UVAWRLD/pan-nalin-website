# Remove room splats that would sit in front of the boy (inside his outline,
# nearer than his surface at that pixel), so nothing from the room covers him.
import sys
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.ndimage import maximum_filter, binary_dilation
F, CX, CY, OX, OY, S = 5268.6147, 1885.5, 1285.0, 69, 161, 4
W, H = 3840 // S, 2731 // S
def load(p):
    v = PlyData.read(p)["vertex"]; return v, np.stack([v["x"], v["y"], v["z"]], 1)
# usage: python clear_room.py [room distance] [room .ply] [output .ply]
ROOM_IN = sys.argv[2] if len(sys.argv) > 2 else "sharp_out_mj/room_mj.ply"
ROOM_OUT = sys.argv[3] if len(sys.argv) > 3 else "room_mj_clear.ply"
room_v, R = load(ROOM_IN)
# The site can draw the room closer or farther (SETTINGS.room.distance scales
# it around the camera); judge "in front of the boy" at that distance.
ROOM_DISTANCE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
R_seen = R * ROOM_DISTANCE
_, B = load("boy_splat.ply")
pix = lambda P: ((F * P[:, 0] / P[:, 2] + CX + OX) / S, (F * P[:, 1] / P[:, 2] + CY + OY) / S)
bu, bv = pix(B)
near = np.full((H, W), np.inf)              # the boy's nearest surface per pixel
ok = (bu >= 0) & (bu < W) & (bv >= 0) & (bv < H)
np.minimum.at(near, (bv[ok].astype(int), bu[ok].astype(int)), B[ok, 2])
near = -maximum_filter(-near, 5)            # widen it a little (min filter)
cover = binary_dilation(np.isfinite(near), iterations=4)
ru, rv = pix(R_seen)
iu = np.clip(ru.astype(int), 0, W - 1); iv = np.clip(rv.astype(int), 0, H - 1)
inside = (ru >= 0) & (ru < W) & (rv >= 0) & (rv < H) & cover[iv, iu]
# Only what is in front of his nearest surface: the floor under him and the
# wall behind him stay, so moving the camera shows room, not a black hole.
front_of_boy = np.where(np.isfinite(near[iv, iu]), near[iv, iu], -1)
drop = inside & (R_seen[:, 2] < front_of_boy - 0.03)
print("dropped", drop.sum(), "room splats in front of the boy")
PlyData([PlyElement.describe(room_v.data[~drop], "vertex")]).write(ROOM_OUT)
