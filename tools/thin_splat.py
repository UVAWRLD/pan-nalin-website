# Thins a single-view splat (e.g. SHARP output) for the web by merging
# neighbours that land in the same small screen cell and depth slice, as seen
# from the camera it was made from (origin, OpenCV axes).
#
#   python thin_splat.py room.ply room_thin.ply --focal 1976 --cell 1.5
#
# --focal is the focal length in pixels at the display size the cell is
# measured in (1976 = the photo camera shown 1440 px wide). Merged splats take
# the opacity-weighted mean position and colour, the combined opacity, and
# scales grown to cover the area of the splats they replace.

import argparse
import numpy as np
from plyfile import PlyData, PlyElement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--focal", type=float, default=1976)
    ap.add_argument("--cell", type=float, default=1.5, help="screen cell, px")
    ap.add_argument("--depth-step", type=float, default=0.03, help="relative depth slice")
    ap.add_argument("--max-grow", type=float, default=0, help="cap on how much a merged splat grows (x)")
    args = ap.parse_args()

    v = PlyData.read(args.src)["vertex"]
    names = v.data.dtype.names
    d = {k: np.asarray(v[k], np.float64) for k in names}
    x, y, z = d["x"], d["y"], d["z"]
    a = 1 / (1 + np.exp(-d["opacity"]))

    u = np.floor(args.focal * x / z / args.cell).astype(np.int64)
    w = np.floor(args.focal * y / z / args.cell).astype(np.int64)
    s = np.floor(np.log(z) / np.log1p(args.depth_step)).astype(np.int64)
    key = (u - u.min()) * 10**12 + (w - w.min()) * 10**6 + (s - s.min())
    _, inv, count = np.unique(key, return_inverse=True, return_counts=True)
    m = len(count)

    wsum = np.bincount(inv, a, m) + 1e-9
    mean = lambda k: np.bincount(inv, d[k] * a, m) / wsum
    out = {}
    for k in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"):
        out[k] = mean(k)
    # combined opacity of the group: 1 - prod(1 - a)
    keep_a = 1 - np.exp(np.bincount(inv, np.log1p(-np.clip(a, 0, 0.999)), m))
    out["opacity"] = np.log(keep_a / (1 - keep_a + 1e-6) + 1e-9)
    # rotation and shape from the most opaque member, scaled up to cover the group
    order = np.lexsort((-a, inv))
    first = order[np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1]]
    # Grow only as much as needed to cover the merged area: by the square
    # root of the count, but never past the cell size on screen (big growth
    # turns many tiny splats into soft blobs).
    grow = np.log(np.sqrt(count))
    if args.max_grow:
        grow = np.minimum(grow, np.log(args.max_grow))
    for k in ("scale_0", "scale_1", "scale_2"):
        out[k] = d[k][first] + grow
    for k in ("rot_0", "rot_1", "rot_2", "rot_3"):
        out[k] = d[k][first]
    for k in ("nx", "ny", "nz"):
        out[k] = np.zeros(m)

    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    el = np.empty(m, dtype=[(f, "f4") for f in fields])
    for f in fields:
        el[f] = out[f]
    PlyData([PlyElement.describe(el, "vertex")]).write(args.dst)
    print(f"{len(x)} -> {m} splats")


if __name__ == "__main__":
    main()
