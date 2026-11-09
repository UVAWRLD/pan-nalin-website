# Converts a 3D Gaussian splat .ply (degree-0 colours) to .spz, the compressed
# format the site loads with Spark. About 10x smaller than the .ply.
#
#   python ply_to_spz.py room.ply room.spz [--max 800000] [--crop-depth 12]
#
# SPZ v2 layout (Niantic spz): 16-byte header, then gzip of positions
# (24-bit fixed point), alphas, colours, scales, rotations.

import argparse, gzip, struct
import numpy as np
from plyfile import PlyData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max", type=int, default=0, help="keep at most this many splats (most opaque first)")
    ap.add_argument("--crop-depth", type=float, default=0, help="drop splats farther than this along z")
    args = ap.parse_args()

    v = PlyData.read(args.src)["vertex"]
    pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
    op = np.asarray(v["opacity"], np.float32)
    sc = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)  # w x y z

    alpha = 1 / (1 + np.exp(-op))
    keep = alpha > 1 / 255
    if args.crop_depth:
        keep &= pos[:, 2] < args.crop_depth
    idx = np.where(keep)[0]
    if args.max and len(idx) > args.max:
        idx = idx[np.argsort(-alpha[idx])[: args.max]]
    pos, dc, alpha, sc, rot = pos[idx], dc[idx], alpha[idx], sc[idx], rot[idx]
    n = len(pos)

    frac = 12
    lim = 2 ** 23 - 1
    fixed = np.clip(np.round(pos * (1 << frac)), -lim, lim).astype(np.int32).ravel()
    pos_bytes = np.stack([fixed & 0xFF, (fixed >> 8) & 0xFF, (fixed >> 16) & 0xFF], 1).astype(np.uint8)

    u8 = lambda x: np.clip(np.round(x), 0, 255).astype(np.uint8)
    alphas = u8(alpha * 255)
    colors = u8(dc * (0.15 * 255) + 0.5 * 255)
    scales = u8((sc + 10) * 16)
    q = rot / np.linalg.norm(rot, axis=1, keepdims=True)
    q *= np.where(q[:, :1] < 0, -1, 1)                      # w >= 0
    rots = u8(q[:, 1:] * 127.5 + 127.5)                       # x y z

    header = struct.pack("<IIIBBBB", 0x5053474E, 2, n, 0, frac, 0, 0)
    body = b"".join(a.tobytes() for a in (pos_bytes, alphas, colors, scales, rots))
    with open(args.dst, "wb") as f:
        f.write(gzip.compress(header + body, 9))
    print(f"wrote {args.dst}: {n} splats")


if __name__ == "__main__":
    main()
