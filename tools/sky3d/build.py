# Builds the "boys and the sky" 3D scene (assets/sky/) from the film still in
# tools/sky3d/src/:
#   boys_wide.webp          the still (2000x1125)
#   boys_wide_cutout.webp   the six boys cut out (transparency)
#   boys_tall.webp          a taller version of the same moment, with more sky above
#
# The scene, drawn by scene3d.js (the same viewer as the other hero scene):
#   backdrop   sky, far shore and lake, extended well beyond the photo's frame
#              (the photo, the taller version above it, and an AI fill for the
#              rest), placed far behind the boys
#   boy_1..6   each boy on his own layer, with the relief of his body from a
#              depth model, at his own distance: boy 3 (the long-haired one)
#              nearest, boy 4 (the small one, behind him) farthest, the others
#              in a row between
#
#   .venv/bin/python tools/sky3d/build.py
# Slow steps keep their results in work/sky/ and are reused.
import json, subprocess, sys
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools/scene3d"))
from common import ell, nearest_fill, lama, laplace_fill

WORK = ROOT / "work/sky"
WORK.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "assets/sky"
OUT.mkdir(parents=True, exist_ok=True)
SRC = HERE / "src"
PY = sys.executable

C = cv2.imread(str(SRC / "boys_wide.webp")).astype(np.float32)
H, W = C.shape[:2]
CUT = cv2.imread(str(SRC / "boys_wide_cutout.webp"), cv2.IMREAD_UNCHANGED)
A_all = CUT[..., 3].astype(np.float32) / 255
TALL = cv2.imread(str(SRC / "boys_tall.webp")).astype(np.float32)
TALL_CUT = cv2.imread(str(SRC / "boys_tall_cutout.webp"), cv2.IMREAD_UNCHANGED)
# how far the backdrop continues past the photo's frame (photo px)
PAD_L, PAD_R, PAD_T, PAD_B = 900, 900, 1300, 160


def cached_img(name, fn):
    p = WORK / f"{name}.png"
    if p.exists():
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32)
    img = fn()
    cv2.imwrite(str(p), np.clip(img, 0, 255).astype(np.uint8))
    return img.astype(np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ---- 1. the six boys, one mask each ------------------------------------------
# A box and a point or two per boy (photo px), left to right; the segmentation
# model separates the ones that overlap (3 with 4, 5 with 6).
BOYS = {
    "boy_1": {"box": [65, 575, 360, 1125], "pos": [[200, 750], [190, 950]]},
    "boy_2": {"box": [540, 575, 810, 1125], "pos": [[650, 850], [660, 1050]]},
    "boy_3": {"box": [765, 500, 1210, 1125], "pos": [[1000, 850], [1060, 620], [960, 1000]]},
    "boy_4": {"box": [1135, 740, 1375, 1125], "pos": [[1260, 920], [1250, 790]]},
    "boy_5": {"box": [1335, 600, 1660, 1125], "pos": [[1500, 870], [1490, 1050]]},
    "boy_6": {"box": [1605, 575, 1895, 1125], "pos": [[1790, 870], [1790, 1050]]},
}
seg_dir = WORK / "seg"
if not all((seg_dir / f"{n}.png").exists() for n in BOYS):
    k = 1920 / W                                           # segment.py takes points in a 1920-wide frame
    prompts = {n: {"box": [v * k for v in b["box"]], "pos": [[x * k, y * k] for x, y in b["pos"]]} for n, b in BOYS.items()}
    json.dump(prompts, open(WORK / "prompts.json", "w"))
    subprocess.run([PY, str(ROOT / "tools/scene3d/segment.py"), str(SRC / "boys_wide.webp"), str(WORK / "prompts.json"),
                    str(seg_dir)], check=True)
seg = {n: cv2.imread(str(seg_dir / f"{n}.png"), 0) > 0 for n in BOYS}
# every cut-out pixel belongs to exactly one boy: the one whose mask holds it
# (the nearer boy where two do), else the nearest mask
ORDER_NEAR_FIRST = ["boy_3", "boy_1", "boy_2", "boy_5", "boy_6", "boy_4"]
owner = np.full((H, W), -1, np.int32)
for i, n in enumerate(ORDER_NEAR_FIRST):
    owner[(owner < 0) & seg[n]] = i
inside = A_all > 0.02
if (inside & (owner < 0)).any():
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(owner < 0, return_distances=False, return_indices=True)
    owner = np.where(owner < 0, owner[idx[0], idx[1]], owner)
alphas = {}
for i, n in enumerate(ORDER_NEAR_FIRST):
    m = cv2.GaussianBlur((owner == i).astype(np.float32), (0, 0), 0.8)
    alphas[n] = (A_all * m).astype(np.float32)

# ---- 2. depth ---------------------------------------------------------------
dz = WORK / "depth.npz"
if not dz.exists():
    subprocess.run([PY, str(ROOT / "tools/scene3d/depth.py"), str(SRC / "boys_wide.webp"), str(dz)], check=True)
dd = np.load(dz)
dp, focal = dd["depth"].astype(np.float32), float(dd["focal_px"])

# ---- 3. the backdrop: boys removed, extended beyond the frame ----------------
EW, EH = W + PAD_L + PAD_R, H + PAD_T + PAD_B
boys_wide = cv2.dilate((A_all > 0.02).astype(np.uint8), ell(10)) > 0


def clean_plate():
    return lama(np.clip(C, 0, 255).astype(np.uint8), boys_wide, max_side=2000, pad=200)


PLATE = cached_img("plate", clean_plate)


def tall_in_wide():
    """The taller version, brought into the wide photo's frame (it holds real sky above)."""
    ta = TALL_CUT[..., 3]
    wa = CUT[..., 3]
    g1 = cv2.cvtColor(TALL.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(C.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(6000)
    k1, d1 = orb.detectAndCompute(g1, (ta > 128).astype(np.uint8) * 255)
    k2, d2 = orb.detectAndCompute(g2, (wa > 128).astype(np.uint8) * 255)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    src = np.float32([k1[x.queryIdx].pt for x in m])
    dst = np.float32([k2[x.trainIdx].pt for x in m]) + np.float32([PAD_L, PAD_T])
    Mx, inl = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=4, maxIters=5000)
    print("tall -> wide: %d matches, %d agree, scale %.3f" % (len(m), int(inl.sum()), float(np.hypot(Mx[0, 0], Mx[0, 1]))))
    tall_clean = lama(TALL.astype(np.uint8), cv2.dilate((ta > 5).astype(np.uint8), ell(12)) > 0, max_side=2000, pad=200)
    img = cv2.warpAffine(tall_clean.astype(np.float32), Mx, (EW, EH), flags=cv2.INTER_CUBIC)
    cov = cv2.warpAffine(np.ones(TALL.shape[:2], np.float32), Mx, (EW, EH), flags=cv2.INTER_LINEAR)
    cov = cv2.erode((cov > 0.99).astype(np.uint8), ell(4))
    return np.dstack([img, cov * 255])


TW = cached_img("tall_in_wide", tall_in_wide)
SKY_PROMPT = ("photograph, film still, dreamy summer sky with soft cumulus clouds, warm golden sunlight, "
              "pastel blue and cream colours, a calm lake and a low line of green trees on the far horizon, "
              "soft focus, gentle haze")
SKY_NEG = ("people, person, boy, body, hands, text, letters, watermark, frame, border, buildings, birds, "
           "sharp detail, illustration, painting")


def backdrop():
    from outpaint import outpaint
    canvas = np.zeros((EH, EW, 3), np.float32)
    known = np.zeros((EH, EW), bool)
    tw_ok = TW[..., 3] > 127
    canvas[tw_ok] = TW[..., :3][tw_ok]
    known |= tw_ok
    canvas[PAD_T:PAD_T + H, PAD_L:PAD_L + W] = PLATE
    known[PAD_T:PAD_T + H, PAD_L:PAD_L + W] = True
    # behind the boys the photo never saw anything: let the fill redo it too
    known[PAD_T:PAD_T + H, PAD_L:PAD_L + W] &= ~(cv2.dilate(boys_wide.astype(np.uint8), ell(6)) > 0)
    Q = 2
    qs = (EW // Q, EH // Q)
    cq = np.clip(cv2.resize(canvas, qs, interpolation=cv2.INTER_AREA), 0, 255).astype(np.uint8)
    kq = cv2.resize(known.astype(np.float32), qs, interpolation=cv2.INTER_AREA) > 0.99
    base = lama(cq, ~kq, max_side=1600, pad=4000, feather=1)
    painted = outpaint(cq, kq, laplace_fill, init=base, strength=0.55, band=8, steps=40, guidance=5.5,
                       prompt=SKY_PROMPT, negative=SKY_NEG)
    up = cv2.resize(painted.astype(np.float32), (EW, EH), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 1.0)
    w = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 4)[..., None]
    return canvas * w + up * (1 - w)


BACK = cached_img("backdrop", backdrop)

# ---- 4. place everything --------------------------------------------------------
cx, cy = W / 2, H / 2
med = {n: float(np.median(dp[alphas[n] > 0.9])) for n in BOYS}
print("depth model, boys (m):", {n: round(v, 2) for n, v in med.items()})
# Long lens: the boys stand a few metres away and close together. Keep the
# model's spacing for the row, but put 3 clearly nearest and 4 clearly behind.
D = float(np.median(list(med.values())))
row = {n: D + np.clip(med[n] - D, -0.3, 0.3) for n in BOYS}
row["boy_3"] = min(row.values()) - 0.7
row["boy_4"] = max(row.values()) + 0.7
R = D * 6.0                                                 # the sky, shore and lake, far behind
print("placed (m):", {n: round(v, 2) for n, v in row.items()}, "backdrop", round(R, 1))

layers = []


def add_layer(name, rgb, a, depth, rect, cell, opaque=False):
    q = [cv2.IMWRITE_WEBP_QUALITY, 95]
    P = np.clip(rgb * (1 if a is None else a[..., None]), 0, 255).astype(np.uint8)
    for tag, k in (("", 1), ("@half", 2)):
        sz = (max(1, P.shape[1] // k), max(1, P.shape[0] // k))
        cv2.imwrite(str(OUT / f"{name}{tag}.webp"), P if k == 1 else cv2.resize(P, sz, interpolation=cv2.INTER_AREA), q)
        if a is not None:
            al = np.round(a * 255).astype(np.uint8)
            cv2.imwrite(str(OUT / f"{name}{tag}-alpha.png"), al if k == 1 else cv2.resize(al, sz, interpolation=cv2.INTER_AREA))
    h, w = depth.shape
    step = cell
    gw, gh = -(-w // step) + 1, -(-h // step) + 1
    ds = cv2.GaussianBlur(depth, (0, 0), step / 3)
    g = ds[np.ix_(np.minimum(np.arange(gh) * step, h - 1), np.minimum(np.arange(gw) * step, w - 1))]
    inv = 1.0 / g
    lo, hi = float(inv.min()), float(inv.max())
    qd = np.round((inv - lo) / max(hi - lo, 1e-9) * 65535).astype(np.uint16)
    img = np.zeros(qd.shape + (3,), np.uint8)
    img[..., 2], img[..., 1] = qd >> 8, qd & 255
    cv2.imwrite(str(OUT / f"{name}-depth.png"), img)
    L = {"name": name, "color": f"{name}.webp", "color_half": f"{name}@half.webp", "depth": f"{name}-depth.png",
         "opaque": opaque, "inv_min": lo, "inv_max": hi, "grid": [gw, gh], "rect": rect, "cell": float(step)}
    if a is not None:
        L["alpha"], L["alpha_half"] = f"{name}-alpha.png", f"{name}@half-alpha.png"
    layers.append(L)


# the backdrop, a little further out at its edges than in the middle (a gentle
# curve, so a turned camera does not see it as a flat wall), its border
# continued so the camera never runs off it
EXT = 900
bk = cv2.copyMakeBorder(BACK, EXT, EXT, EXT, EXT, cv2.BORDER_REPLICATE)
bk = cv2.GaussianBlur(bk, (0, 0), 0.6)
by, bx = np.mgrid[0:bk.shape[0], 0:bk.shape[1]].astype(np.float32)
ux = (bx - EXT - PAD_L - cx) / focal
d_back = (R * (1 + 0.25 * ux * ux)).astype(np.float32)
Q = 2
add_layer("backdrop", cv2.resize(bk, (bk.shape[1] // Q, bk.shape[0] // Q), interpolation=cv2.INTER_AREA), None,
          cv2.resize(d_back, (bk.shape[1] // Q, bk.shape[0] // Q), interpolation=cv2.INTER_AREA),
          [-PAD_L - EXT, -PAD_T - EXT, bk.shape[1], bk.shape[0]], 8, opaque=True)
# each layer's cell is in its own image's pixels; the viewer wants photo px per cell
layers[-1]["cell"] = 8.0 * Q

# the boys, far to near
for n in sorted(BOYS, key=lambda n: -row[n]):
    a = alphas[n]
    ys, xs = np.where(a > 0.01)
    x0, x1 = max(0, xs.min() - 12), min(W, xs.max() + 13)
    y0, y1 = max(0, ys.min() - 12), min(H, ys.max() + 13)
    core = cv2.erode((a > 0.9).astype(np.uint8), ell(3)) > 0
    rel = np.clip(dp - med[n], -0.35, 0.35) * 0.8              # the relief of his body
    rel = np.where(core, rel, nearest_fill(rel, core))
    rel = cv2.GaussianBlur(rel, (0, 0), 3)
    depth = (row[n] + rel)[y0:y1, x0:x1]
    colour = np.where((a > 0.9)[..., None], C, cv2.GaussianBlur(nearest_fill(C, core), (0, 0), 1.5))
    add_layer(n, colour[y0:y1, x0:x1], a[y0:y1, x0:x1], depth.astype(np.float32), [int(x0), int(y0), int(x1 - x0), int(y1 - y0)], 3)

scene = {"width": W, "height": H, "focal": focal, "cx": cx, "cy": cy, "pivot_depth": D, "layers": layers}
json.dump(scene, open(OUT / "scene.json", "w"), indent=1)
print("written", OUT / "scene.json")
