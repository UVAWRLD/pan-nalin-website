# Third version of the "boys and the sky" scene (assets/sky3/), drawn by sky3d.js.
# Everything comes from one seamless panorama of the moment (pano.jpg, 15924 x
# 7600): the sky dome and the lake are that one picture, so there are no joins.
#
#   sky dome   the panorama with the boys painted out, on a sphere around the camera
#   lake       a real water surface showing the panorama's water, rippling
#   boys       each on his own relief (body shape from Sapiens), picture at full
#              resolution; where one boy stands behind another, the hidden edge
#              of the one behind is continued a little, so nothing is missing
#              when they slide apart
#
# Inputs in tools/sky3d/src/: pano.jpg (the panorama), wide.png and
# wide_cutout.png (the same moment, 2880 px, and the boys cut out of it).
# Reuses work/sky/seg (each boy's mask) and work/sky/depth.npz from build.py.
#   .venv/bin/python tools/sky3d/build3.py
import json, sys
from pathlib import Path
import numpy as np
import cv2
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools/scene3d"))
from common import ell, nearest_fill, lama, smooth_fill

WORK = ROOT / "work/sky"
OUT = ROOT / "assets/sky3"
OUT.mkdir(parents=True, exist_ok=True)
SRC = HERE / "src"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
BOYS = ["boy_1", "boy_2", "boy_3", "boy_4", "boy_5", "boy_6"]
NEAR_FIRST = ["boy_3", "boy_1", "boy_2", "boy_5", "boy_6", "boy_4"]
K = 2                                     # working copy at half the panorama's resolution

# ---- the panorama, at half size (and the full size kept for the boys' pictures) ----
half_p = WORK / "pano_half.png"
PANO_FULL = cv2.imread(str(SRC / "pano.jpg"), cv2.IMREAD_COLOR)
if not half_p.exists():
    cv2.imwrite(str(half_p), cv2.resize(PANO_FULL, (PANO_FULL.shape[1] // K, PANO_FULL.shape[0] // K), interpolation=cv2.INTER_AREA))
P = cv2.imread(str(half_p)).astype(np.float32)
H3, W3 = P.shape[:2]

# ---- where the wide photo sits in the panorama ---------------------------------------
WIDE = cv2.imread(str(SRC / "wide.png"), cv2.IMREAD_COLOR)
CUT = cv2.imread(str(SRC / "wide_cutout.png"), cv2.IMREAD_UNCHANGED)
Hw, Ww = WIDE.shape[:2]
reg_p = WORK / "wide_to_pano.npy"
if not reg_p.exists():
    g1 = cv2.cvtColor(WIDE, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(P.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(12000)
    k1, d1 = orb.detectAndCompute(g1, (CUT[..., 3] > 128).astype(np.uint8) * 255)
    k2, d2 = orb.detectAndCompute(g2, None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    src = np.float32([k1[x.queryIdx].pt for x in m]); dst = np.float32([k2[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=3, maxIters=10000)
    print("wide -> panorama: %d matches, %d agree" % (len(m), int(inl.sum())))
    np.save(reg_p, M)
M = np.load(reg_p)
s = float(np.hypot(M[0, 0], M[0, 1]))
warp = lambda img, interp=cv2.INTER_LINEAR: cv2.warpAffine(img, M, (W3, H3), flags=interp)
A_all = warp(CUT[..., 3].astype(np.float32) / 255)
# the earlier per-boy masks were made on the 2000 px copy of the wide photo
S2000 = np.array([[Ww / 2000, 0, 0], [0, Ww / 2000, 0], [0, 0, 1]])
M2000 = (np.vstack([M, [0, 0, 1]]) @ S2000)[:2]
warp2000 = lambda img, interp=cv2.INTER_LINEAR: cv2.warpAffine(img, M2000, (W3, H3), flags=interp)
seg = {n: warp2000((cv2.imread(str(WORK / "seg" / f"{n}.png"), 0) > 0).astype(np.uint8), cv2.INTER_NEAREST) > 0 for n in BOYS}
dd = np.load(WORK / "depth.npz")
focal = float(dd["focal_px"]) * float(np.hypot(*M2000[0, :2]))   # the photo's focal length, in panorama px
cx, cy = M2000 @ np.array([1000.0, 1125 / 2, 1.0])                # the photo's centre, in panorama px
dp = warp2000(dd["depth"].astype(np.float32))
print("focal %.0f px, centre (%.0f, %.0f), panorama %d x %d (half size)" % (focal, cx, cy, W3, H3))

# ---- the six boys ---------------------------------------------------------------------
owner = np.full((H3, W3), -1, np.int32)
for i, n in enumerate(NEAR_FIRST):
    owner[(owner < 0) & seg[n]] = i
inside = A_all > 0.02
if (inside & (owner < 0)).any():
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(owner < 0, return_distances=False, return_indices=True)
    owner = np.where(owner < 0, owner[idx[0], idx[1]], owner)
alphas = {n: (A_all * cv2.GaussianBlur((owner == i).astype(np.float32), (0, 0), 1.0)).astype(np.float32)
          for i, n in enumerate(NEAR_FIRST)}
# The panorama reaches further down than the cut-out (more lake at the bottom,
# and the boys' shorts carry on to the edge): continue each boy down to the
# bottom edge, column by column, wherever the picture is not the blue lake.
lake = (P[..., 0] - P[..., 2] > 20) & (P.max(2) > 90)
for n in BOYS:
    a = alphas[n]
    rows = np.where((a > 0.5).sum(1) > 10)[0]
    yb = int(rows.max())
    if yb < H3 - 2:
        tmpl = a[yb - 3:yb + 1].max(0)
        keep = ~cv2.morphologyEx(lake.astype(np.uint8), cv2.MORPH_OPEN, ell(3)).astype(bool)
        for y in range(yb + 1, H3):
            a[y] = np.maximum(a[y], tmpl * keep[y])
        a[yb + 1:] = cv2.GaussianBlur(a[yb + 1:], (0, 0), 1.0)
A_all = np.maximum.reduce([alphas[n] for n in BOYS])

# ---- body shape per boy (Sapiens) -------------------------------------------------------
sap_p = WORK / "sapiens3.npz"
if not sap_p.exists():
    from huggingface_hub import hf_hub_download
    net = torch.jit.load(hf_hub_download("facebook/sapiens-depth-1b-torchscript",
                                         "sapiens_1b_render_people_epoch_88_torchscript.pt2"), map_location="cpu").eval().to(DEV)
    mean = torch.tensor([123.5, 116.5, 103.5]).view(1, 3, 1, 1)
    std = torch.tensor([58.5, 57.0, 57.5]).view(1, 3, 1, 1)
    res = {}
    for n in BOYS:
        ys, xs = np.where(alphas[n] > 0.5)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        h = max(y1 - y0, (x1 - x0) * 4 / 3) * 1.1
        w = h * 3 / 4
        bx0, by0, bw, bh = int((x0 + x1) / 2 - w / 2), int((y0 + y1) / 2 - h / 2), int(w), int(h)
        pad = cv2.copyMakeBorder(P.astype(np.uint8), bh, bh, bw, bw, cv2.BORDER_REPLICATE)
        crop = pad[by0 + bh:by0 + 2 * bh, bx0 + bw:bx0 + 2 * bw]
        inp = cv2.resize(crop, (768, 1024), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
        t = (torch.from_numpy(inp).permute(2, 0, 1)[None].float() - mean) / std
        with torch.no_grad():
            o = net(t.to(DEV)).float().cpu()[0, 0].numpy()
        res[n + "_box"] = np.array([bx0, by0, bw, bh])
        res[n + "_d"] = cv2.resize(o, (bw, bh), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        print("sapiens", n, flush=True)
    np.savez(sap_p, **res)
SAP = np.load(sap_p)

# ---- distances --------------------------------------------------------------------------
med = {n: float(np.median(dp[alphas[n] > 0.9])) for n in BOYS}
D = float(np.median(list(med.values())))
row = {n: D + float(np.clip(med[n] - D, -0.3, 0.3)) for n in BOYS}
row["boy_3"] = min(row.values()) - 0.7
row["boy_4"] = max(row.values()) + 0.5
row["boy_6"] = min(row["boy_6"], row["boy_5"] - 0.3)       # his raised elbow passes in front of boy 5's hands
print("placed (m):", {n: round(v, 2) for n, v in row.items()})

# ---- the sky: the panorama with the boys painted out ----------------------------------------
sky_p = WORK / "pano_sky.png"
if not sky_p.exists():
    cv2.imwrite(str(sky_p), lama(P.astype(np.uint8), cv2.dilate((A_all > 0.02).astype(np.uint8), ell(12)) > 0,
                                 max_side=2048, pad=300))
SKY = cv2.imread(str(sky_p))
q = [cv2.IMWRITE_WEBP_QUALITY, 92]
cv2.imwrite(str(OUT / "sky.webp"), SKY, q)
cv2.imwrite(str(OUT / "sky@half.webp"), cv2.resize(SKY, (W3 // 2, H3 // 2), interpolation=cv2.INTER_AREA), q)

# the horizon: the top of the lake beside the boys
blue = (P[..., 0] - P[..., 2]) > 20
free = A_all < 0.02
rows_w = [y for y in range(H3 // 2, H3) if (blue[y] & free[y]).sum() > 0.5 * max(free[y].sum(), 1)]
horizon = float(min(rows_w)) if rows_w else H3 * 0.9
print("horizon row", horizon)
# the light: the brightest part of the sky, well above the lake
g = cv2.GaussianBlur(cv2.cvtColor(SKY, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), W3 / 60)
g[int(horizon * 0.75):] = 0
sy, sx = np.unravel_index(np.argmax(g), g.shape)
sun = [float(sx), float(sy)]
print("sun at", sun)

# ---- the boys' layers -------------------------------------------------------------------------
layers = []
CELL = 2                                                  # half-panorama px per depth cell
ys, xs = np.where(A_all > 0.02)
group = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
far_first = sorted(BOYS, key=lambda n: -row[n])
for n in far_first:
    a = alphas[n].copy()
    # The part of him hidden behind nearer boys: continue his outline a little
    # way under them (40 px), so a sliver of him, not of nothing, shows when
    # they slide apart.
    nearer = np.zeros((H3, W3), np.float32)
    for m_ in BOYS:
        if row[m_] < row[n]:
            nearer = np.maximum(nearer, alphas[m_])
    hid = (nearer > 0.3) & (cv2.dilate((a > 0.5).astype(np.uint8), ell(40)) > 0)
    if hid.any():
        cont = np.clip(smooth_fill(a, ~hid), 0, 1)
        cont = cv2.GaussianBlur((cv2.GaussianBlur(cont, (0, 0), 3) > 0.5).astype(np.float32), (0, 0), 1.2)
        a = np.where(hid, np.maximum(a, cont), a)
    elbow = None
    if n == "boy_4":
        # His raised arm goes behind boy 3, and the photo never shows his
        # elbow: a painting of it (tools/sky3d/src/boy4_elbow.png, chosen by
        # eye), of which only the skin is kept.
        ex, ey = 4250, 3110
        paint = cv2.imread(str(SRC / "boy4_elbow.png")).astype(np.float32)
        emask = np.load(SRC / "boy4_elbow_mask.npy")
        es = paint.shape[0]
        sky_c = cv2.imread(str(WORK / "pano_sky.png"))[ey:ey + es, ex:ex + es].astype(np.float32)
        skin = (paint[..., 2] - paint[..., 0] > 25) & (paint[..., 2] > 80)
        differs = np.abs(paint - sky_c).max(2) > 14
        e_a = cv2.morphologyEx((emask & skin & differs).astype(np.uint8), cv2.MORPH_OPEN, ell(2))
        e_a = cv2.GaussianBlur(e_a.astype(np.float32), (0, 0), 1.2) * emask
        a[ey:ey + es, ex:ex + es] = np.maximum(a[ey:ey + es, ex:ex + es], e_a)
        elbow = (ex, ey, paint, e_a)
    a[a < 0.01] = 0
    core = cv2.erode((alphas[n] > 0.9).astype(np.uint8), ell(2)) > 0
    # body shape from Sapiens, its sign and scale set against the depth model
    bx0, by0, bw, bh = [int(v) for v in SAP[n + "_box"]]
    full = np.zeros((H3, W3), np.float32); have = np.zeros((H3, W3), bool)
    ya, xa, yb, xb = max(0, by0), max(0, bx0), min(H3, by0 + bh), min(W3, bx0 + bw)
    full[ya:yb, xa:xb] = SAP[n + "_d"][ya - by0:yb - by0, xa - bx0:xb - bx0]; have[ya:yb, xa:xb] = True
    c2 = core & have
    sign = 1.0 if np.corrcoef(full[c2], dp[c2])[0, 1] >= 0 else -1.0
    rel = sign * (full - np.median(full[c2]))
    rel = rel / (np.percentile(np.abs(rel[c2]), 95) + 1e-6) * 0.15
    rel = cv2.GaussianBlur(np.where(c2, rel, nearest_fill(rel, c2)), (0, 0), 1.2)
    depth = row[n] + np.clip(rel, -0.35, 0.35)
    # soft shadow from the boys in front of him, on what they hide (and a touch beyond)
    shade = 1.0 - 0.35 * cv2.GaussianBlur(nearer, (0, 0), 12) * (cv2.dilate((nearer > 0.05).astype(np.uint8), ell(6)) > 0)
    yy, xx = np.where(a > 0.01)
    x0, x1, y0, y1 = max(0, xx.min() - 8), min(W3, xx.max() + 9), max(0, yy.min() - 8), min(H3, yy.max() + 9)
    # picture: full resolution; at the soft edge and under nearer boys, his own colour continued
    A2 = cv2.resize(a[y0:y1, x0:x1], ((x1 - x0) * K, (y1 - y0) * K), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    col = PANO_FULL[y0 * K:y1 * K, x0 * K:x1 * K].astype(np.float32)
    own = cv2.resize(core[y0:y1, x0:x1].astype(np.uint8), (A2.shape[1], A2.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    own &= cv2.resize((nearer[y0:y1, x0:x1] < 0.05).astype(np.uint8), (A2.shape[1], A2.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    col = np.where(own[..., None], col, cv2.GaussianBlur(nearest_fill(col, own), (0, 0), 2.0))
    if elbow is not None:
        ex, ey, paint, e_a = elbow
        big = cv2.resize(paint, (paint.shape[1] * K, paint.shape[0] * K), interpolation=cv2.INTER_CUBIC)
        wa = cv2.resize(e_a, (paint.shape[1] * K, paint.shape[0] * K), interpolation=cv2.INTER_LINEAR)[..., None]
        oy, ox = (ey - y0) * K, (ex - x0) * K                    # where the painting sits in his picture
        ya, xa = max(0, oy), max(0, ox)
        yb, xb = min(col.shape[0], oy + big.shape[0]), min(col.shape[1], ox + big.shape[1])
        if yb > ya and xb > xa:
            seg_ = (slice(ya, yb), slice(xa, xb)); bseg = (slice(ya - oy, yb - oy), slice(xa - ox, xb - ox))
            col[seg_] = col[seg_] * (1 - wa[bseg]) + big[bseg] * wa[bseg]
    if y1 >= H3:                                          # cut by the picture's bottom edge: carry him on below it
        ext = 160
        A2 = np.pad(A2, ((0, ext * K), (0, 0)), mode="edge")
        col = np.pad(col, ((0, ext * K), (0, 0), (0, 0)), mode="edge")
        depth = np.pad(depth, ((0, ext), (0, 0)), mode="edge")
        shade = np.pad(shade, ((0, ext), (0, 0)), mode="edge")
        y1 += ext
    rgb = np.clip(col * A2[..., None], 0, 255).astype(np.uint8)
    al = np.round(A2 * 255).astype(np.uint8)
    for tag, k in (("", 1), ("@half", 2)):
        sz = (max(1, rgb.shape[1] // k), max(1, rgb.shape[0] // k))
        cv2.imwrite(str(OUT / f"{n}{tag}.webp"), rgb if k == 1 else cv2.resize(rgb, sz, interpolation=cv2.INTER_AREA), q)
        cv2.imwrite(str(OUT / f"{n}{tag}-alpha.png"), al if k == 1 else cv2.resize(al, sz, interpolation=cv2.INTER_AREA))
    dg = cv2.resize(depth[y0:y1, x0:x1], (-(-(x1 - x0) // CELL) + 1, -(-(y1 - y0) // CELL) + 1), interpolation=cv2.INTER_AREA)
    inv = 1.0 / dg
    lo, hi = float(inv.min()), float(inv.max())
    qd = np.round((inv - lo) / max(hi - lo, 1e-9) * 65535).astype(np.uint16)
    img = np.zeros(qd.shape + (3,), np.uint8); img[..., 2], img[..., 1] = qd >> 8, qd & 255
    img[..., 0] = np.round(cv2.resize(shade[y0:y1, x0:x1], (qd.shape[1], qd.shape[0]), interpolation=cv2.INTER_AREA) * 255).clip(0, 255)
    cv2.imwrite(str(OUT / f"{n}-depth.png"), img)
    layers.append({"name": n, "color": f"{n}.webp", "color_half": f"{n}@half.webp", "alpha": f"{n}-alpha.png",
                   "alpha_half": f"{n}@half-alpha.png", "depth": f"{n}-depth.png", "inv_min": lo, "inv_max": hi,
                   "grid": [int(qd.shape[1]), int(qd.shape[0])], "rect": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                   "cell": float(CELL)})
    print(n, "at %.2f m" % row[n])

scene = {"width": W3, "height": H3, "focal": focal, "cx": float(cx), "cy": float(cy), "pivot_depth": D,
         "sky": {"image": "sky.webp", "image_half": "sky@half.webp", "rect": [0, 0, W3, H3]},
         "horizon": horizon, "sun": sun, "camera_height": 1.0, "group": group, "layers": layers}
json.dump(scene, open(OUT / "scene.json", "w"), indent=1)
print("written", OUT / "scene.json")
