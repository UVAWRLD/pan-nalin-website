# Second version of the "boys and the sky" scene (assets/sky2/), drawn by sky3d.js.
#
#   sky dome   the photo's own sky, clouds and far shore (extended beyond the
#              frame by build.py), on a sphere around the camera: seamless, at
#              infinity, the clouds drifting slowly
#   lake       a real horizontal water surface (the photo's water, rippling)
#   boys       each on his own relief, its shape from Sapiens (Meta's model for
#              people: fine body depth), the picture upscaled 2x (Real-ESRGAN)
#
# Needs build.py's results in work/sky/ (separate boys, depth, extended backdrop).
#   .venv/bin/python tools/sky3d/build2.py
import json, sys
from pathlib import Path
import numpy as np
import cv2
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools/scene3d"))
from common import ell, nearest_fill

WORK = ROOT / "work/sky"
OUT = ROOT / "assets/sky2"
OUT.mkdir(parents=True, exist_ok=True)
SRC = HERE / "src"
UP = 2                                                    # upscale factor for the boys
PAD_L, PAD_R, PAD_T, PAD_B = 900, 900, 1300, 160           # as in build.py
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

C = cv2.imread(str(SRC / "boys_wide.webp"))
H, W = C.shape[:2]
A_all = cv2.imread(str(SRC / "boys_wide_cutout.webp"), cv2.IMREAD_UNCHANGED)[..., 3].astype(np.float32) / 255
dd = np.load(WORK / "depth.npz")
dp, focal = dd["depth"].astype(np.float32), float(dd["focal_px"])
BOYS = ["boy_1", "boy_2", "boy_3", "boy_4", "boy_5", "boy_6"]
NEAR_FIRST = ["boy_3", "boy_1", "boy_2", "boy_5", "boy_6", "boy_4"]


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ---- the six boys' masks (as build.py) ----------------------------------------
seg = {n: cv2.imread(str(WORK / "seg" / f"{n}.png"), 0) > 0 for n in BOYS}
owner = np.full((H, W), -1, np.int32)
for i, n in enumerate(NEAR_FIRST):
    owner[(owner < 0) & seg[n]] = i
if ((A_all > 0.02) & (owner < 0)).any():
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(owner < 0, return_distances=False, return_indices=True)
    owner = np.where(owner < 0, owner[idx[0], idx[1]], owner)

# ---- 1. upscale 2x (Real-ESRGAN), kept gentle -----------------------------------
up_p = WORK / "wide_x2.png"
if not up_p.exists():
    from huggingface_hub import hf_hub_download
    from spandrel import ModelLoader
    model = ModelLoader().load_from_file(hf_hub_download("ai-forever/Real-ESRGAN", "RealESRGAN_x2.pth")).model.eval().to(DEV)
    x = torch.from_numpy(C[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255
    out = torch.zeros(1, 3, H * 2, W * 2)
    T, O = 384, 32
    with torch.no_grad():
        for y0 in range(0, H, T):
            for x0 in range(0, W, T):
                ya, xa = max(0, y0 - O), max(0, x0 - O)
                yb, xb = min(H, y0 + T + O), min(W, x0 + T + O)
                r = model(x[:, :, ya:yb, xa:xb].to(DEV)).cpu()
                oy, ox = (y0 - ya) * 2, (x0 - xa) * 2
                hh, ww = (min(H, y0 + T) - y0) * 2, (min(W, x0 + T) - x0) * 2
                out[:, :, y0 * 2:y0 * 2 + hh, x0 * 2:x0 * 2 + ww] = r[:, :, oy:oy + hh, ox:ox + ww]
    sr = (out[0].permute(1, 2, 0).numpy()[:, :, ::-1].clip(0, 1) * 255).astype(np.float32)
    # gentle: keep the photo's own tones and grain, take only the finer detail from the upscaler
    plain = cv2.resize(C, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    detail = sr - cv2.GaussianBlur(sr, (0, 0), 1.6)
    res = cv2.GaussianBlur(plain, (0, 0), 1.6) + 0.85 * detail
    cv2.imwrite(str(up_p), np.clip(res, 0, 255).astype(np.uint8))
C2 = cv2.imread(str(up_p)).astype(np.float32)

# ---- 2. body depth per boy (Sapiens) ------------------------------------------------
sap_p = WORK / "sapiens_depth.npz"
if not sap_p.exists():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("facebook/sapiens-depth-1b-torchscript", "sapiens_1b_render_people_epoch_88_torchscript.pt2")
    net = torch.jit.load(path, map_location="cpu").eval()
    dev = DEV
    try:
        net = net.to(dev)
    except Exception:
        dev = "cpu"
    mean = torch.tensor([123.5, 116.5, 103.5]).view(1, 3, 1, 1)
    std = torch.tensor([58.5, 57.0, 57.5]).view(1, 3, 1, 1)
    sap = {}
    for n in BOYS:
        ys, xs = np.where(owner == NEAR_FIRST.index(n))
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        # a 3:4 (w:h) crop around him, a little margin
        cxb, cyb = (x0 + x1) / 2, (y0 + y1) / 2
        h = max(y1 - y0, (x1 - x0) * 4 / 3) * 1.1
        w = h * 3 / 4
        bx0, by0 = int(round(cxb - w / 2)), int(round(cyb - h / 2))
        bw, bh = int(round(w)), int(round(h))
        crop = cv2.copyMakeBorder(C, max(0, -by0), max(0, by0 + bh - H), max(0, -bx0), max(0, bx0 + bw - W),
                                  cv2.BORDER_REPLICATE)[max(0, by0):max(0, by0) + bh, max(0, bx0):max(0, bx0) + bw]
        inp = cv2.resize(crop, (768, 1024), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
        t = (torch.from_numpy(inp).permute(2, 0, 1)[None].float() - mean) / std
        with torch.no_grad():
            try:
                o = net(t.to(dev)).float().cpu()
            except Exception:
                net, dev = net.to("cpu"), "cpu"
                o = net(t).float().cpu()
        d = cv2.resize(o[0, 0].numpy(), (bw, bh), interpolation=cv2.INTER_CUBIC)
        sap[n] = np.array([bx0, by0, bw, bh], np.int64), d.astype(np.float32)
        print("sapiens", n, "done on", dev, flush=True)
    np.savez(sap_p, **{f"{n}_box": b for n, (b, d) in sap.items()}, **{f"{n}_d": d for n, (b, d) in sap.items()})
S = np.load(sap_p)

# ---- 3. place the boys -----------------------------------------------------------------
med_dp = {}
for n in BOYS:
    m = (owner == NEAR_FIRST.index(n)) & (A_all > 0.9)
    med_dp[n] = float(np.median(dp[m]))
D = float(np.median(list(med_dp.values())))
row = {n: D + float(np.clip(med_dp[n] - D, -0.3, 0.3)) for n in BOYS}
row["boy_3"] = min(row.values()) - 0.7
row["boy_4"] = max(row.values()) + 0.7

layers = []


def write_layer(name, P, a, depth, rect, cell_px):
    """P straight colour, a alpha, depth (m) on the same grid; rect in photo px; cell_px photo px per depth cell."""
    q = [cv2.IMWRITE_WEBP_QUALITY, 95]
    rgb = np.clip(P * a[..., None], 0, 255).astype(np.uint8)
    al = np.round(a * 255).astype(np.uint8)
    for tag, k in (("", 1), ("@half", 2)):
        sz = (max(1, rgb.shape[1] // k), max(1, rgb.shape[0] // k))
        cv2.imwrite(str(OUT / f"{name}{tag}.webp"), rgb if k == 1 else cv2.resize(rgb, sz, interpolation=cv2.INTER_AREA), q)
        cv2.imwrite(str(OUT / f"{name}{tag}-alpha.png"), al if k == 1 else cv2.resize(al, sz, interpolation=cv2.INTER_AREA))
    inv = 1.0 / depth
    lo, hi = float(inv.min()), float(inv.max())
    qd = np.round((inv - lo) / max(hi - lo, 1e-9) * 65535).astype(np.uint16)
    img = np.zeros(qd.shape + (3,), np.uint8)
    img[..., 2], img[..., 1] = qd >> 8, qd & 255
    cv2.imwrite(str(OUT / f"{name}-depth.png"), img)
    layers.append({"name": name, "color": f"{name}.webp", "color_half": f"{name}@half.webp",
                   "alpha": f"{name}-alpha.png", "alpha_half": f"{name}@half-alpha.png", "depth": f"{name}-depth.png",
                   "inv_min": lo, "inv_max": hi, "grid": [int(qd.shape[1]), int(qd.shape[0])],
                   "rect": [float(v) for v in rect], "cell": float(cell_px)})


CELL = 2                                                  # photo px per depth cell for the boys
A2 = cv2.resize(A_all, (W * UP, H * UP), interpolation=cv2.INTER_CUBIC).clip(0, 1)
for n in sorted(BOYS, key=lambda n: -row[n]):
    mine = (owner == NEAR_FIRST.index(n)).astype(np.float32)
    a = A_all * cv2.GaussianBlur(mine, (0, 0), 0.8)
    box, sd = S[f"{n}_box"], S[f"{n}_d"]
    bx0, by0, bw, bh = [int(v) for v in box]
    full = np.zeros((H, W), np.float32)
    have = np.zeros((H, W), bool)
    ya, xa = max(0, by0), max(0, bx0)
    yb, xb = min(H, by0 + bh), min(W, bx0 + bw)
    full[ya:yb, xa:xb] = sd[ya - by0:yb - by0, xa - bx0:xb - bx0]
    have[ya:yb, xa:xb] = True
    core = (a > 0.9) & have
    # Sapiens gives relative depth; set its sign and scale against the metric
    # depth model inside his body, then keep the body's shape at a plausible
    # thickness (about 30 cm front to back)
    s_in, d_in = full[core], dp[core]
    sign = 1.0 if np.corrcoef(s_in, d_in)[0, 1] >= 0 else -1.0
    rel = sign * (full - np.median(s_in))
    spread = np.percentile(np.abs(rel[core]), 95) + 1e-6
    rel = rel / spread * 0.15
    rel = np.where(core, rel, nearest_fill(rel, core))
    rel = cv2.GaussianBlur(rel, (0, 0), 1.2)
    depth = row[n] + np.clip(rel, -0.35, 0.35)
    ys, xs = np.where(a > 0.01)
    x0, x1 = max(0, xs.min() - 8), min(W, xs.max() + 9)
    y0, y1 = max(0, ys.min() - 8), min(H, ys.max() + 9)
    # picture at 2x, its colour at the soft edge taken from inside him (no sky halo)
    a2 = cv2.resize(a, (W * UP, H * UP), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    core2 = cv2.erode((a2 > 0.9).astype(np.uint8), ell(3)) > 0
    col = np.where((a2 > 0.9)[..., None], C2, cv2.GaussianBlur(nearest_fill(C2, core2), (0, 0), 1.5))
    sl2 = (slice(y0 * UP, y1 * UP), slice(x0 * UP, x1 * UP))
    dgrid = depth[y0:y1, x0:x1]
    dgrid = cv2.resize(dgrid, (-(-(x1 - x0) // CELL) + 1, -(-(y1 - y0) // CELL) + 1), interpolation=cv2.INTER_AREA)
    write_layer(n, col[sl2], a2[sl2], dgrid.astype(np.float32), [x0, y0, x1 - x0, y1 - y0], CELL)
    print(n, "at %.2f m" % row[n])

# ---- 4. the sky dome and the lake's picture --------------------------------------------
BACK = cv2.imread(str(WORK / "backdrop.png")).astype(np.float32)   # photo frame + extension, boys removed
cv2.imwrite(str(OUT / "sky.webp"), np.clip(BACK, 0, 255).astype(np.uint8), [cv2.IMWRITE_WEBP_QUALITY, 92])
# where the lake meets the far shore (the horizon), in photo rows
g = cv2.cvtColor(C, cv2.COLOR_BGR2GRAY).astype(np.float32)
blue = (C[..., 0].astype(np.float32) - C[..., 2]) > 20
cols = (A_all < 0.02)
rows_water = [y for y in range(H // 2, H) if (blue[y] & cols[y]).sum() > 0.5 * max(cols[y].sum(), 1)]
horizon = float(min(rows_water)) if rows_water else H * 0.9
print("horizon row", horizon)

scene = {"width": W, "height": H, "focal": focal, "cx": W / 2, "cy": H / 2, "pivot_depth": D,
         "sky": {"image": "sky.webp", "rect": [-PAD_L, -PAD_T, W + PAD_L + PAD_R, H + PAD_T + PAD_B]},
         "horizon": horizon, "camera_height": 1.0,
         # framing: desktop shows the photo's frame; tall screens show the sky above (as the tall version does)
         "frames": {"wide": [0, 0, W, H], "tall": [0, -1215, W, H + 1215]},
         "layers": layers}
json.dump(scene, open(OUT / "scene.json", "w"), indent=1)
print("written", OUT / "scene.json")
