# Sharp old room in the middle, wide room only around it. Both are in the
# photo camera's frame and at the same scale (wide was depth-matched).
import sys
import numpy as np
from plyfile import PlyData, PlyElement
F, CX, CY, OX, OY = 5268.6147, 1885.5, 1285.0, 69, 161
INSET = 0.03   # drop the old room's outermost 3%: SHARP stretches its edge splats
def pix(P): return F * P[:, 0] / P[:, 2] + CX + OX, F * P[:, 1] / P[:, 2] + CY + OY
old = PlyData.read(sys.argv[2] if len(sys.argv) > 2 else "sharp_out_mj/room_mj.ply")["vertex"].data
import sys
wide = PlyData.read(sys.argv[1] if len(sys.argv) > 1 else "room_wide_scaled.ply")["vertex"].data
x0, y0, x1, y1 = 3840 * INSET, 2731 * INSET, 3840 * (1 - INSET), 2731 * (1 - INSET)
ou, ov = pix(np.stack([old["x"], old["y"], old["z"]], 1))
wu, wv = pix(np.stack([wide["x"], wide["y"], wide["z"]], 1))
keep_old = (ou > x0) & (ou < x1) & (ov > y0) & (ov < y1)
keep_wide = ~((wu > x0) & (wu < x1) & (wv > y0) & (wv < y1))
names = old.dtype.names
merged = np.concatenate([old[keep_old][list(names)], wide[keep_wide][list(names)]])
print("old kept", keep_old.sum(), "wide kept", keep_wide.sum(), "total", len(merged))
PlyData([PlyElement.describe(merged, "vertex")]).write("room_merged.ply")
