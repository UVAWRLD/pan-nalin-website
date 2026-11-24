# Builds the layered 3D scene the page renders (assets/scene/).
#
# The photo is split into layers, far to near:
#   surround  the wall and floor, continued a little beyond the photo's frame (half resolution)
#   far       the wall and floor inside the frame
#   objects   the trunk and film cans
#   film_back the parts of the film strip behind the boy (the blurred swing, the strand by the trunk)
#   boy       the boy, without the film and without his near hand
#   film_front the parts of the film strip in front of him (the loop and the strands down his body)
#   hand      his near (left) forearm and hand, which grips the strip
# Each layer is a colour image with transparency plus a depth map; the page
# turns each into a relief mesh seen from the photo's camera. Behind every
# layer the layers below are complete (painted in where the photo never saw
# them), so moving the camera uncovers picture, not holes.
#
# Colours are stored premultiplied and solved bottom-up so that, from the
# photo's own viewpoint, the layers add up to the photo exactly:
#     layer = photo - (1 - alpha) * everything_behind
#
#   python tools/scene3d/build.py
#
# Inputs (work/ is made by tools/scene3d/prepare.py; run that first):
#   assets/img/hero-boy-film-strip.png   the photo
#   tools/voyager/room_clean.png         the room without the boy
#   work/depth_photo.npz, work/depth_room_aligned.npy, work/alpha_fg.png, work/alpha_obj.png
import json, subprocess, sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from common import ell, nearest_fill, smooth_fill, lama

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
CACHE = WORK / "cache"
CACHE.mkdir(exist_ok=True)
OUT = ROOT / "assets" / "scene"
OUT.mkdir(parents=True, exist_ok=True)
STEP = 6                                   # one depth-map cell (and mesh vertex) every STEP photo pixels
PAD_L, PAD_R, PAD_T, PAD_B = 648, 648, 336, 228   # how far the room continues past the frame, px

C = cv2.imread(str(ROOT / "assets/img/hero-boy-film-strip.png")).astype(np.float32)
B = cv2.imread(str(ROOT / "tools/voyager/room_clean.png")).astype(np.float32)
H, W = C.shape[:2]
dnpz = np.load(WORK / "depth_photo.npz")
dp = dnpz["depth"]
focal = float(dnpz["focal_px"])
dra = np.load(WORK / "depth_room_aligned.npy")
a_fg = cv2.imread(str(WORK / "alpha_fg.png"), 0).astype(np.float32) / 255
a_fg[a_fg < 0.012] = 0
# Inside the loop of film between his hands the wall shows through. The matte
# is hazy there; where the depth map says "wall", make it properly empty.
LOOP_HOLE = np.array([(1262, 650), (1420, 612), (1590, 640), (1640, 760), (1600, 800), (1560, 1000),
                      (1470, 1020), (1300, 860), (1245, 780)], np.int32)
loop_hole = np.zeros((H, W), np.uint8)
cv2.fillPoly(loop_hole, [LOOP_HOLE], 1)
loop_hole = (loop_hole > 0) & (a_fg < 0.85) & ((dra - dp) < 0.3)
loop_hole = cv2.morphologyEx(loop_hole.astype(np.uint8), cv2.MORPH_OPEN, ell(2)) > 0
loop_soft = np.minimum(cv2.GaussianBlur(cv2.erode(loop_hole.astype(np.uint8), ell(6)).astype(np.float32), (0, 0), 8) * 1.6, 1)
a_fg = (a_fg * (1 - loop_soft)).astype(np.float32)
a_fg[a_fg < 0.012] = 0
a_vit = a_fg.copy()                                         # the matte as the model gave it
a_obj = cv2.imread(str(WORK / "alpha_obj.png"), 0).astype(np.float32) / 255
a_obj[a_obj < 0.012] = 0
# The cans' right edge, which runs down beside the raised arm, came out ragged
# from the cut-out; replace it with a smooth fit to it, so no slivers of wall
# flicker through its notches when the arm moves.
rows = np.arange(720, 1290)
edge = np.array([np.where(a_obj[y, :1300] > 0.5)[0].max() for y in rows], np.float32)
fit = np.polyval(np.polyfit(rows, edge, 2), rows)
for y, e in zip(rows, fit):
    xs = np.arange(int(e) - 45, int(e) + 46)
    a_obj[y, xs] = np.clip(0.5 - (xs - e) / 2.5, 0, 1)


def cached(name, fn):
    p = CACHE / f"{name}.png"
    if p.exists():
        return cv2.imread(str(p)).astype(np.float32)
    img = fn()
    cv2.imwrite(str(p), np.clip(img, 0, 255).astype(np.uint8))
    return img.astype(np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def cliffs(d, size=7, jump=0.03):
    k = np.ones((size, size), np.uint8)
    return (cv2.dilate(d, k) - cv2.erode(d, k)) > jump


def guided(mask, guide_bgr, r=8, eps=6.0):
    """Edge-aware smoothing of a mask so its border follows the picture's edges."""
    I = cv2.cvtColor(guide_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    p = mask.astype(np.float32) * 255
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps * eps)
    b = mp - a * mI
    return np.clip((box(a) * I + box(b)) / 255, 0, 1)


def laplace_fill(values, fixed, cond=None):
    """Solve for the unknown cells of `values` (h x w or h x w x c) so that each is
    the weighted average of its four neighbours, with the `fixed` cells held:
    the smoothest surface that passes through what is known. `cond` (h x w)
    makes it stiffer where it is larger."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    h, w = fixed.shape
    cond = np.ones((h, w)) if cond is None else cond
    idx = np.arange(h * w).reshape(h, w)
    rows, cols, vals = [], [], []
    diag = np.zeros(h * w)
    for (sa, sb) in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                     ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
        i, j = idx[sa].ravel(), idx[sb].ravel()
        wt = np.minimum(cond[sa], cond[sb]).ravel()
        rows += [i, j]; cols += [j, i]; vals += [-wt, -wt]
        np.add.at(diag, i, wt); np.add.at(diag, j, wt)
    L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w, h * w))
    L = L + sp.diags(diag)
    f = fixed.ravel()
    free = ~f
    lu = splu(L[free][:, free].tocsc())
    Lff = L[free][:, f]
    v = values.reshape(h * w, -1).astype(np.float64).copy()
    for c in range(v.shape[1]):
        v[free, c] = lu.solve(-(Lff @ v[f, c]))
    return v.reshape(values.shape).astype(np.float32)


def extend(d, core, alpha, S=6):
    """Continue depth beyond `core` (where it is trusted) over the whole image.

    Wherever the layer shows anything (alpha > 0) the depth is held stiff, so a
    soft rim or a veil of blur sits flat with the thing it belongs to; the
    change from one thing's depth to another's happens in the empty gaps
    between them.
    """
    h, w = d.shape[0] // S, d.shape[1] // S
    cw = cv2.resize(core.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    dv = cv2.resize(np.where(core, d, 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) / np.maximum(cw, 1e-6)
    al = cv2.resize(alpha.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    x = laplace_fill(dv, cw > 0.5, 1.0 + 400.0 * np.clip(al * 4, 0, 1))
    up = cv2.resize(x, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_LINEAR)
    # the first few pixels off the core take the core's own value, then ease into the solution
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
    t = smoothstep(3, 20, dist)
    out = nearest_fill(d, core) * (1 - t) + up * t
    soft = cv2.GaussianBlur(out, (0, 0), 3)
    return np.where(core, d, soft).astype(np.float32)


def layer_depth(d, solid, alpha, erode=7, jump=0.03, tol=0.10):
    """Depth for one layer: trusted inside the layer away from its edges, continued outside."""
    core = cv2.erode(solid.astype(np.uint8), ell(erode)) > 0
    core &= ~cv2.dilate(cliffs(d, 7, jump).astype(np.uint8), ell(2)).astype(bool)
    # drop pixels whose depth disagrees with the layer around them (the depth
    # map's outline never matches the matte's exactly), then fill again
    ref = cv2.GaussianBlur(smooth_fill(d, core), (0, 0), 20)
    core &= np.abs(d - ref) < tol
    return extend(d, core, alpha)


# His left side (viewer's left) runs as a smooth curve from the armpit to the
# waist; where the hanging strip crosses it, continue that curve (fitted to
# the rows where the edge is clearly visible) rather than the cut-out's guess.
gray_ = cv2.cvtColor(np.clip(C, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
fit_y, fit_x = [], []
for y in list(range(1230, 1561, 5)) + list(range(1670, 1861, 5)):
    run = 0
    for i, v in enumerate(gray_[y, 1550:1850] > 70):
        run = run + 1 if v else 0
        if run >= 40:
            fit_y.append(y); fit_x.append(1550 + i - 39); break
edge_c = np.polyfit(np.array(fit_y, np.float64), np.array(fit_x, np.float64), 3)
ey = np.arange(1150, 1990, dtype=np.float64)
edge_x = np.polyval(edge_c, np.clip(ey, 1230, 1890))
side = np.zeros((H, W), np.float32)
xs_ = np.arange(1500, 1950, dtype=np.float32)[None, :]
side[1150:1990, 1500:1950] = np.clip((xs_ - edge_x[:, None].astype(np.float32)) / 3 + 0.5, 0, 1)
side[1150:1990, 1950:] = 1
side_w = np.zeros((H, W), np.float32)
side_w[1230:1960, 1500:1950] = 1
side_w = cv2.GaussianBlur(side_w, (0, 0), 10)
# The dark gap beside his flank (left of that side, right of the trunk) is the
# room in shadow, not him: the wall layer shows it, and the wall behind him is
# continued from it, so it never turns into a hard-edged dark shape.
_strip_band = np.zeros((H, W), np.uint8)
cv2.polylines(_strip_band, [np.array([(1520, 970), (1535, 1100), (1555, 1300), (1580, 1400), (1625, 1490), (1680, 1560),
                                      (1745, 1630), (1805, 1690), (1860, 1760), (1925, 1875)], np.int32)], False, 1, 130)
_yy, _xx = np.mgrid[0:H, 0:W]
_aobj0 = cv2.imread(str(WORK / "alpha_obj.png"), 0) > 128
_gray0 = cv2.cvtColor(np.clip(C, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
_lit_boy = cv2.dilate(((cv2.imread(str(WORK / "alpha_fg.png"), 0) > 128) & (_gray0 > 45)).astype(np.uint8), ell(10)) > 0
_brown0 = np.zeros((H, W), np.uint8)
cv2.fillPoly(_brown0, [np.array([(1660, 1000), (1880, 1000), (1880, 1560), (1780, 1560)], np.int32)], 1)
gap_wall = (side < 0.05) & (_yy >= 1180) & (_yy < 1900) & (_xx > 1100) & ~_aobj0 & (_gray0 < 45) & \
           (_strip_band == 0) & ~_lit_boy & (_brown0 == 0)
gap_wall &= cv2.erode((~_aobj0).astype(np.uint8), ell(6)) > 0
gap_wall = cv2.morphologyEx(gap_wall.astype(np.uint8), cv2.MORPH_OPEN, ell(3)) > 0

# ---- the motion-blurred film in the top-left --------------------------------
# It is a soft dark veil over the wall. The room plate still carries a faint
# ghost of it. So: paint the ghost out of the plate for its overall tone, read
# how much darker the photo is than that (the veil's opacity, low frequencies
# only), and divide the veil out of the photo itself. The wall keeps its own
# texture, the veil becomes a smooth transparent layer, and the two still add
# up to the photo.
veil_zone = cv2.dilate((a_fg > 0).astype(np.uint8), ell(80)) > 0
_arc = np.zeros((H, W), bool)
_arc[330:770, 0:1300] = True                                 # the whole blurred arc, above the cans
veil_zone |= _arc
veil_zone[:, 1300:] = False
veil_zone[760:] = False
veil_zone &= cv2.dilate((a_obj > 0).astype(np.uint8), ell(12)) == 0
B_tone = cached("plate_clean", lambda: lama(np.clip(B, 0, 255).astype(np.uint8), veil_zone, max_side=960, pad=350))


def masked_blur(x, m, sigma):
    m = m.astype(np.float32)
    num = cv2.GaussianBlur(x * (m[..., None] if x.ndim == 3 else m), (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den[..., None] if x.ndim == 3 else den, 1e-4)


see_through = a_fg < 0.3                                     # not the hand or the solid film
film = np.array([14.0, 18.0, 26.0], np.float32)              # the blurred film's own colour (B, G, R): near black, warm
lum = lambda x: x @ np.array([0.114, 0.587, 0.299], np.float32)
ratio = masked_blur(lum(C - film), see_through, 14) / np.maximum(masked_blur(lum(B_tone - film), see_through, 14), 1.0)
# weight that reaches zero well inside the zone's border, so nothing changes abruptly there
inner = cv2.GaussianBlur((cv2.erode(veil_zone.astype(np.uint8), ell(34)) > 0).astype(np.float32), (0, 0), 10)
veil = np.clip(1 - ratio, 0, 0.9) * inner * veil_zone
veil[veil < 0.02] = 0
wall = film + (C - film) / (1 - veil[..., None])
# Where the film is dense, dividing it out amplifies noise into black: there
# the painted plate is the wall. The divided wall is also kept near the
# plate's tone, so it can sharpen texture but never invent dark shapes.
wall = np.clip(wall, B_tone * 0.95 - 2, B_tone * 1.3 + 6)      # the film's darkening goes with the film, not into the wall
t = smoothstep(0.22, 0.45, veil)[..., None]
wall = wall * (1 - t) + B_tone * t
mix = (smoothstep(0.3, 0.6, a_fg) * 1.0)[..., None]          # under solid things keep the painted plate
B = np.where(veil_zone[..., None], np.clip(wall * (1 - mix) + B_tone * mix, 0, 255), B).astype(np.float32)
# (the blurred film itself is part of the film strip, built further down)

# ---- leftover outlines in the plate -------------------------------------------
# Where the plate was painted, its mask ran a little inside the boy's soft
# edge, so a thin rim of him is still in it. Repaint every pixel that belongs
# to the foreground but which the plate left as it was in the photo.
def clean_rims():
    same = np.abs(cv2.GaussianBlur(B, (0, 0), 1.5) - cv2.GaussianBlur(C, (0, 0), 1.5)).max(2) < 6
    rim = (cv2.dilate((a_vit > 0.04).astype(np.uint8), ell(8)) > 0) & same & ~veil_zone
    rim[2300:] = False                                       # film lying on the floor in front stays part of the floor
    # in the blur zone the wall was rebuilt from the photo, rim and all: repaint
    # a band around everything solid there (the hand, the arm, the loop of film)
    rim |= veil_zone & (cv2.dilate((a_vit > 0.25).astype(np.uint8), ell(14)) > 0)
    rim &= a_vit > 0                                         # pixels the matte calls pure background are the real wall: keep them
    rim = cv2.morphologyEx(rim.astype(np.uint8), cv2.MORPH_OPEN, ell(1)) > 0
    out = np.clip(B, 0, 255).astype(np.uint8)
    T = 1300
    for y0 in range(0, H, T):
        for x0 in range(0, W, T):
            m = np.zeros((H, W), bool)
            m[y0:y0 + T, x0:x0 + T] = rim[y0:y0 + T, x0:x0 + T]
            if m.any():
                out = lama(out, m, max_side=960, pad=160)    # worked small: the wall is out of focus, and the model draws a dot grid when worked large
    return out


B = cached("plate_rims", clean_rims)

# ============ the empty room: depth =========================================
known = cv2.dilate((a_fg > 0).astype(np.uint8), ell(40)) == 0
ratio = smooth_fill(np.where(known, dp / dra, 1).astype(np.float32), known)
ratio = cv2.GaussianBlur(ratio, (0, 0), 25)
d_room = (dra * ratio).astype(np.float32)

# ============ the room beyond the frame (half resolution) =====================
EW, EH = W + PAD_L + PAD_R, H + PAD_T + PAD_B


def pad_img(img, mode=cv2.BORDER_REPLICATE):
    return cv2.copyMakeBorder(img, PAD_T, PAD_B, PAD_L, PAD_R, mode)


def outpaint_room():
    """The room (with trunk) continued past the frame. First a plain inpainting
    gives the right tone and layout; then a diffusion model, run gently on top,
    gives it the texture of real plaster and stone. Painted at a quarter of the
    photo's resolution (it is out of focus anyway) and enlarged."""
    from outpaint import outpaint
    Q = 4
    qw, qh = EW // Q, EH // Q
    canvas = np.clip(cv2.resize(pad_img(B), (qw, qh), interpolation=cv2.INTER_AREA), 0, 255).astype(np.uint8)
    known = np.zeros((qh, qw), bool)
    known[PAD_T // Q + 1:(PAD_T + H) // Q - 1, PAD_L // Q + 1:(PAD_L + W) // Q - 1] = True
    def paint():
        base = lama(canvas, ~known, max_side=1408, pad=4000, feather=1)
        return outpaint(canvas, known, laplace_fill, init=base, strength=0.42, band=10, steps=48, guidance=5.0)
    painted = cached("room_ext_painted", paint)                 # the slow part, kept on its own
    half = cv2.resize(painted, (EW // 2, EH // 2), interpolation=cv2.INTER_CUBIC)
    real = cv2.resize(pad_img(B), (EW // 2, EH // 2), interpolation=cv2.INTER_AREA)
    # the real photo inside the frame, blending into the painted room over a wide band
    inside = np.zeros((EH // 2, EW // 2), np.float32)
    inside[PAD_T // 2:(PAD_T + H) // 2, PAD_L // 2:(PAD_L + W) // 2] = 1
    d = cv2.distanceTransform((inside > 0).astype(np.uint8), cv2.DIST_L2, 5)
    w = smoothstep(4, 24, d)[..., None]
    # The painted room drifts in colour and is crisper than the out-of-focus
    # photo: carry the photo's own tone out past the edge (fading with
    # distance), soften it to the photo's focus, and blend over a wide band.
    edge_band = (inside > 0) & (d < 50)
    shift = laplace_fill(cv2.resize(np.where(edge_band[..., None], cv2.GaussianBlur(real - half, (0, 0), 6), 0),
                                    (EW // 8, EH // 8), interpolation=cv2.INTER_AREA),
                         cv2.resize(edge_band.astype(np.float32), (EW // 8, EH // 8), interpolation=cv2.INTER_AREA) > 0.5)
    shift = cv2.GaussianBlur(cv2.resize(shift, (EW // 2, EH // 2), interpolation=cv2.INTER_LINEAR), (0, 0), 10)
    out_d = cv2.distanceTransform((inside == 0).astype(np.uint8), cv2.DIST_L2, 5)
    half = cv2.GaussianBlur(half, (0, 0), 1.6) + shift * np.exp(-out_d / 260)[..., None]
    w = smoothstep(2, 60, d)[..., None]
    return real * w + half * (1 - w)


room_half = outpaint_room()                                    # room with trunk, extended
a_obj_ext = pad_img(a_obj)
obj_ext_half = cv2.resize(a_obj_ext, (EW // 2, EH // 2), interpolation=cv2.INTER_AREA)


def far_half():
    m = cv2.dilate((obj_ext_half > 0.02).astype(np.uint8), ell(14)) > 0
    return lama(np.clip(room_half, 0, 255).astype(np.uint8), m, max_side=1024, pad=500)


far_half_img = far_half()                                      # recomputed each time: it follows the room above                # wall and floor only, extended

# full-resolution wall and floor inside the frame
obj_solid = a_obj > 0.5


def far_full():
    return lama(np.clip(B, 0, 255).astype(np.uint8), cv2.dilate((a_obj > 0).astype(np.uint8), ell(24)) > 0, max_side=1024)


FAR = cached("far_full", far_full)


# Just inside the boy's outline the old plate holds a faint ghost of him (a
# lighter rim, streaks). That band is what the camera uncovers when it moves,
# so repaint it as a continuation of the real wall and floor around him.
def far_behind_boy():
    band = (cv2.dilate((a_vit > 0.04).astype(np.uint8), ell(10)) > 0) & \
           ~(cv2.erode((a_vit > 0.5).astype(np.uint8), ell(170)) > 0)
    band &= cv2.dilate((a_obj > 0.3).astype(np.uint8), ell(8)) == 0
    # paint from the real photo wherever nothing stands in front of the wall
    covered = cv2.dilate(((a_fg > 0.02) | (a_obj > 0.02)).astype(np.uint8), ell(4)) > 0
    out = np.clip(np.where(covered[..., None], FAR, C), 0, 255).astype(np.uint8)
    T = 1400
    for y0 in range(0, H, T):
        for x0 in range(0, W, T):
            m = np.zeros((H, W), bool)
            m[y0:y0 + T, x0:x0 + T] = band[y0:y0 + T, x0:x0 + T]
            if m.any():
                out = lama(out, m, max_side=900, pad=200)
    return out


FAR = cached("far_behind_boy", far_behind_boy)

# depth of wall and floor: the room's depth with the trunk bridged over, continued past the frame
obj_wide = cv2.dilate(obj_solid.astype(np.uint8), ell(30)) > 0
d_far = smooth_fill(d_room, ~obj_wide)
d_far = np.maximum(d_far, d_room)
d_far = cv2.GaussianBlur(d_far, (0, 0), 6)


def extend_depth(d):
    rep = pad_img(d)
    ref = pad_img(d, cv2.BORDER_REFLECT_101)
    lin = np.clip(2 * rep - ref, 0.75 * rep, 1.5 * rep)       # carry the slope on past the edge
    m = np.zeros((EH, EW), np.float32)
    m[PAD_T:PAD_T + H, PAD_L:PAD_L + W] = 1
    soft = cv2.GaussianBlur(lin, (0, 0), 20)
    return np.where(m > 0, rep, soft).astype(np.float32)


d_far_ext = extend_depth(d_far)

# ============ objects: trunk and film cans ====================================
# The trunk is a box and the cans a stack of cylinders: give each a clean shape
# fitted to its depth (inverse depth: a plane for the trunk's front, curved
# across for the cans), so neither wobbles or bends where the depth model was
# unsure (behind the arm, at the edges).
def fit_shape(mask, curved):
    core = cv2.erode(mask.astype(np.uint8), ell(25)) > 0
    core &= cv2.dilate((a_fg > 0.2).astype(np.uint8), ell(30)) == 0
    vs, us = np.where(core)
    sel = np.random.default_rng(0).choice(len(us), min(len(us), 40000), replace=False)
    u, v, z = us[sel].astype(np.float64), vs[sel].astype(np.float64), 1 / d_room[vs[sel], us[sel]]
    cols = lambda u, v: np.stack([u, v, np.ones_like(u)] + ([u * u] if curved else []), 1)
    keep = np.ones(len(u), bool)
    for _ in range(4):                                       # trim outliers
        coef = np.linalg.lstsq(cols(u[keep], v[keep]), z[keep], rcond=None)[0]
        r = np.abs(cols(u, v) @ coef - z)
        keep = r < np.percentile(r, 85)
    U, V = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    return (1 / (cols(U.ravel(), V.ravel()) @ coef)).reshape(H, W).astype(np.float32)


cans = cv2.imread(str(WORK / "seg_room/cans.png"), 0) > 0
trunk = cv2.imread(str(WORK / "seg_room/trunk.png"), 0) > 0
d_shape = d_room.copy()
for mask, curved in ((trunk, False), (cans, True)):
    w_ = cv2.GaussianBlur((cv2.dilate(mask.astype(np.uint8), ell(20)) > 0).astype(np.float32), (0, 0), 6)
    d_shape = d_shape * (1 - w_) + fit_shape(mask, curved) * w_
d_obj = layer_depth(d_shape, obj_solid, a_obj, erode=8, jump=0.05)
d_obj = cv2.GaussianBlur(d_obj, (0, 0), 3)
d_obj_ext = pad_img(d_obj)

# Shadows belong to the room. The boy's shadow on the floor and wall is in the
# photo but not in the painted plate behind him, so when the camera moves the
# uncovered floor would be lighter than the shadow around it, leaving a pale
# ghost of his outline. Measure how much darker the photo is than the plate
# just outside him, continue that smoothly underneath him, and darken the plate
# by it, so the shadow carries on behind him and stays on the floor and wall.
lum = lambda x: x @ np.array([0.114, 0.587, 0.299], np.float32)
Q = 4
qs = (W // Q, H // Q)
covered_q = cv2.resize(((cv2.dilate(((a_fg > 0.02) | (a_obj > 0.02)).astype(np.uint8), ell(8)) > 0) & ~gap_wall).astype(np.float32),
                       qs, interpolation=cv2.INTER_AREA)
known_q = covered_q < 0.01
tone = laplace_fill(cv2.resize(C, qs, interpolation=cv2.INTER_AREA), known_q)
tone = cv2.resize(cv2.GaussianBlur(tone, (0, 0), 1.5), (W, H), interpolation=cv2.INTER_CUBIC)
# fine texture: the original plate's (consistent grain), except behind the
# trunk and cans, which the plate still shows; there the painted wall's
obj_cov = (cv2.dilate((a_obj > 0.02).astype(np.uint8), ell(30)) > 0)[..., None]
det_src = np.where(obj_cov, FAR, B)
detail = det_src - cv2.GaussianBlur(det_src, (0, 0), 5)
detail *= obj_cov                                            # behind the boy the plate's texture holds ghosts of his film: none of it
FAR_cont = np.clip(tone + 0.75 * detail, 0, 255)


# Real plaster texture on top of that, from the diffusion painter run gently
# (the wall is out of focus, so a quarter of the photo's resolution is enough),
# blending into the visible wall around the edge so no join shows.
def paint_far_behind():
    from outpaint import outpaint
    canvas = np.clip(cv2.resize(np.where(cv2.resize(covered_q, (W, H)) [..., None] > 0.01, FAR_cont, C), qs,
                                interpolation=cv2.INTER_AREA), 0, 255).astype(np.uint8)
    init = np.clip(cv2.resize(FAR_cont, qs, interpolation=cv2.INTER_AREA), 0, 255).astype(np.uint8)
    return outpaint(canvas, known_q, laplace_fill, init=init, strength=0.5, band=6, steps=40, guidance=5.0, seed=21)


painted_q = cached("far_behind_painted", paint_far_behind).astype(np.float32)
painted = cv2.resize(painted_q, (W, H), interpolation=cv2.INTER_CUBIC)
# keep the painted texture's mid frequencies, the tone fill's overall level
FAR_cont = np.clip(tone + (painted - cv2.GaussianBlur(painted, (0, 0), 24)) + 0.35 * detail, 0, 255)
FAR_cont = cv2.GaussianBlur(FAR_cont, (0, 0), 1.2)              # as out of focus as the real wall
under_far = cv2.GaussianBlur(cv2.resize(covered_q, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), 6)[..., None]
FAR = FAR_cont.astype(np.float32)            # under anything that covers it, only the continued wall: no slivers of the photo

# ============ the film strip, the boy and his near hand =======================
#
# The film is one long strip (tools/scene3d/film_map.png marks it, traced by
# hand). Along its length (letters as on the reference sketch):
#   A, B  blurred, swinging in the air from the left, in front of the wall and
#         over the cans (not lying on them), into
#   C     his raised right hand, which guides it, then
#   D     the loop, arching across to his left hand, which grips it, then
#   E     hanging from that grip just in front of his chest,
#   F     hanging just in front of his raised right arm and his side,
#   G     coming down onto his lap (the top in the air, the bottom resting on it),
#   H     behind his arm and body, down in front of the trunk, landing on his lap.
# Layers, far to near (each later layer is drawn over the earlier ones):
#   film_back   A, B and H: the parts behind him
#   boy         him, without any of the film and without his near hand
#   film_front  C, D, E, F, G
#   hand        his near (left) hand and forearm
film_map = cv2.imread(str(Path(__file__).parent / "film_map.png"), 0) > 127
yy_, xx_ = np.mgrid[0:H, 0:W]
gray = cv2.cvtColor(np.clip(C, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

# which parts of the strip are behind him
ab_zone = (xx_ < 1120) & (yy_ < 900)                                      # A and B
h_zone = (xx_ < 1450) & (yy_ > 1150)                                      # H
back_w = np.clip(cv2.GaussianBlur(np.where(ab_zone | h_zone, 1.0, 0.0).astype(np.float32), (0, 0), 6), 0, 1)

# Just outside the boy's and the trunk's cut-outs the photo still holds bits
# of them the cut-outs missed (a sliver of skin, the trunk's lit top edge).
# On the wall they would stay behind when he moves; give them to their owner.
res = cv2.GaussianBlur(np.abs(cv2.GaussianBlur(C, (0, 0), 3) - cv2.GaussianBlur(FAR_cont, (0, 0), 3)).max(2), (0, 0), 1.5)
strong = smoothstep(18, 34, res)
near_film = cv2.dilate(film_map.astype(np.uint8), ell(24)) > 0
band_b = (cv2.dilate((a_fg > 0.02).astype(np.uint8), ell(28)) > 0) & ~gap_wall & ~near_film & ~ab_zone
band_o = (cv2.dilate((a_obj > 0.02).astype(np.uint8), ell(20)) > 0) & ~band_b & ~near_film
a_fg = np.maximum(a_fg, strong * band_b).astype(np.float32)
a_obj = np.maximum(a_obj, strong * band_o).astype(np.float32)


def run_matte(name, tri):
    """Alpha from the matting model for a trimap, cached until the trimap changes."""
    tri_p, al_p = CACHE / f"trimap_{name}.png", CACHE / f"alpha_{name}.png"
    if not al_p.exists() or not tri_p.exists() or (cv2.imread(str(tri_p), 0) != tri).any():
        cv2.imwrite(str(tri_p), tri)
        subprocess.run([sys.executable, str(Path(__file__).parent / "matte.py"),
                        str(ROOT / "assets/img/hero-boy-film-strip.png"), str(tri_p), str(al_p)], check=True)
    return cv2.imread(str(al_p), 0).astype(np.float32) / 255


# ---- the film: solid parts from the matting model, blurred parts as a veil ----
solid_map = film_map & ~ab_zone
tri = np.zeros((H, W), np.uint8)
tri[cv2.dilate(solid_map.astype(np.uint8), ell(12)) > 0] = 128
tri[cv2.erode(solid_map.astype(np.uint8), ell(4)) > 0] = 255
a_solid_film = run_matte("film", tri) * (cv2.dilate(solid_map.astype(np.uint8), ell(12)) > 0)
# A and B are motion blur: a transparent darkening of whatever is behind them,
# read as how much darker the photo is than the clean wall and lid there.
ab_near = cv2.GaussianBlur((cv2.dilate((film_map & ab_zone).astype(np.uint8), ell(22)) > 0).astype(np.float32), (0, 0), 6)
clean_ab = np.where((a_obj > 0.5)[..., None], cv2.imread(str(ROOT / "tools/voyager/room_clean.png")).astype(np.float32), B_tone)
lum_c = cv2.GaussianBlur(lum(C - film), (0, 0), 3)
lum_p = np.maximum(cv2.GaussianBlur(lum(clean_ab - film), (0, 0), 3), 1.0)
a_ab = np.clip(1 - lum_c / lum_p, 0, 0.9) * ab_near * (xx_ < 1180)
a_ab[a_ab < 0.02] = 0
a_film = np.maximum(a_solid_film, a_ab).astype(np.float32)
a_fb = (a_film * back_w).astype(np.float32)                      # behind him
a_ff = (a_solid_film * (1 - back_w)).astype(np.float32)          # in front of him
a_fb[a_fb < 0.02] = 0
a_ff[a_ff < 0.02] = 0

# ---- his near hand and forearm ---------------------------------------------------
fg_solid = a_fg > 0.6


def front_gap():
    """How far each foreground pixel stands in front of the body around it (metres)."""
    S = 2
    d = cv2.resize(dp, (W // S, H // S), interpolation=cv2.INTER_AREA)
    m = cv2.resize(a_fg, (W // S, H // S), interpolation=cv2.INTER_AREA) > 0.6
    r = 72
    lo = np.where(m, d, -1e3).astype(np.float32)
    dil = cv2.dilate(lo, ell(r))
    hi = np.where(dil < -100, 1e3, dil).astype(np.float32)
    gap = np.where(m, cv2.erode(hi, ell(r)) - d, 0)
    return cv2.resize(gap, (W, H), interpolation=cv2.INTER_LINEAR)


gap_p = CACHE / "front_gap.npy"
if gap_p.exists():
    gap = np.load(gap_p)
else:
    gap = front_gap()
    np.save(gap_p, gap)

ARM = np.array([(1520, 770), (1700, 765), (1800, 790), (1850, 835), (1900, 880), (2000, 975), (2100, 1065),
                (2200, 1105), (2560, 1395), (2580, 1620), (2380, 1620), (2250, 1530), (2000, 1310),
                (1850, 1190), (1700, 1130), (1600, 1090), (1535, 1010), (1515, 900)], np.int32)
arm_roi = np.zeros((H, W), np.uint8)
cv2.fillPoly(arm_roi, [ARM], 1)
hand_hard = ((gap > 0.05) & (arm_roi > 0) & fg_solid & ~(cv2.dilate(film_map.astype(np.uint8), ell(2)) > 0)).astype(np.uint8)
hand_hard = cv2.morphologyEx(hand_hard, cv2.MORPH_OPEN, ell(3))
n_, lab_, st_, _ = cv2.connectedComponentsWithStats(hand_hard, connectivity=8)
for i in range(1, n_):
    if st_[i, cv2.CC_STAT_AREA] < 2500:
        hand_hard[lab_ == i] = 0
hand_hard = cv2.morphologyEx(hand_hard, cv2.MORPH_CLOSE, ell(5)) > 0
tri = np.zeros((H, W), np.uint8)
tri[cv2.dilate(hand_hard.astype(np.uint8), ell(13)) > 0] = 128
tri[cv2.erode(hand_hard.astype(np.uint8), ell(6)) > 0] = 255
a_hand = run_matte("hand", tri) * a_fg
a_hand *= 1 - a_ff                                               # film the hand does not cover stays film
# The hand is out of focus: its outline is soft in the photo, so soft here too.
a_hand = np.maximum(cv2.GaussianBlur(a_hand, (0, 0), 2.2), a_hand * (a_hand > 0.97)).astype(np.float32)
a_hand[a_hand < 0.02] = 0

# ---- him, without the film and without his near hand -----------------------------
cov_all = np.clip(a_film + a_hand, 0, 1)
hole = cv2.dilate((cov_all > 0.05).astype(np.uint8), ell(4)) > 0
a_body_vis = a_fg * (1 - cov_all)
body_fill = np.clip(smooth_fill(a_body_vis, ~hole), 0, 1)
body_sil = cv2.GaussianBlur((cv2.GaussianBlur(body_fill, (0, 0), 4) > 0.5).astype(np.float32), (0, 0), 1.5)
a_boy = np.where(hole, body_sil, a_fg).astype(np.float32)
a_boy *= 1 - cv2.GaussianBlur(gap_wall.astype(np.float32), (0, 0), 2)  # the dark gap beside him is the room
a_boy = a_boy.astype(np.float32)
a_boy[a_boy < 0.02] = 0

# ============ depths ===========================================================
d_boy = layer_depth(dp, (a_boy > 0.6) & ~hole & ~gap_wall, a_boy, erode=7)
d_hand = layer_depth(dp, hand_hard, a_hand, erode=5)


def med(d, mask, default):
    v = d[mask]
    return float(np.median(v)) if v.size > 50 else default


skin_lit = (C[..., 2] > 95) & (C[..., 2] - C[..., 0] > 45)
d_fist = med(dp, (a_vit > 0.95) & (xx_ > 1150) & (xx_ < 1330) & (yy_ > 430) & (yy_ < 700), 1.9)
d_grip = med(d_hand, hand_hard & (xx_ > 1620) & (xx_ < 1780) & (yy_ > 880) & (yy_ < 1060), d_fist)
d_rarm = med(dp, (a_vit > 0.95) & skin_lit & (xx_ > 1000) & (xx_ < 1300) & (yy_ > 900) & (yy_ < 1150), d_fist)
d_chest = med(dp, (a_vit > 0.95) & (xx_ > 1850) & (xx_ < 2050) & (yy_ > 1250) & (yy_ < 1450), 2.1)
d_flank = med(dp, (a_vit > 0.95) & skin_lit & (xx_ > 1680) & (xx_ < 1780) & (yy_ > 1550) & (yy_ < 1750), d_chest)
d_lap = med(dp, (a_vit > 0.95) & (xx_ > 1700) & (xx_ < 2100) & (yy_ > 2050) & (yy_ < 2300), d_chest)
d_knee = med(dp, (a_vit > 0.95) & (xx_ > 1250) & (xx_ < 1450) & (yy_ > 1950) & (yy_ < 2100), d_lap)
d_wall_l = med(d_room, (xx_ > 100) & (xx_ < 600) & (yy_ > 300) & (yy_ < 600), d_fist + 1.0)
d_lid = med(d_obj, (a_obj > 0.9) & (xx_ > 100) & (xx_ < 900) & (yy_ > 700) & (yy_ < 800), d_fist + 0.5)
d_trunk = med(d_obj, (a_obj > 0.9) & (xx_ > 1000) & (xx_ < 1250) & (yy_ > 1300) & (yy_ < 1900), d_flank + 0.4)
d_h = min(d_flank + 0.6 * (d_trunk - d_flank), d_trunk - 0.05)  # behind him, in front of the trunk
print("film anchors (m): fist %.2f grip %.2f right arm %.2f chest %.2f flank %.2f lap %.2f knee %.2f "
      "wall %.2f lid %.2f trunk %.2f H %.2f" % (d_fist, d_grip, d_rarm, d_chest, d_flank, d_lap, d_knee,
                                               d_wall_l, d_lid, d_trunk, d_h))
lerp = lambda a, b, t: a + (b - a) * t
# points along the strip (photo px) and the depth there (m)
FILM_POINTS = {
    "A": [(110, 440, lerp(d_fist, d_wall_l, 0.7)), (320, 380, lerp(d_fist, d_wall_l, 0.6)),
          (580, 420, lerp(d_fist, d_wall_l, 0.45)), (820, 530, lerp(d_fist, d_wall_l, 0.25)),
          (1020, 630, lerp(d_fist, d_wall_l, 0.08))],
    "B": [(110, 640, lerp(d_fist, d_lid - 0.08, 0.7)), (300, 690, lerp(d_fist, d_lid - 0.08, 0.6)),
          (600, 770, lerp(d_fist, d_lid - 0.08, 0.4)), (900, 780, lerp(d_fist, d_lid - 0.08, 0.2)),
          (1080, 760, lerp(d_fist, d_lid - 0.08, 0.05))],
    "C": [(1150, 720, d_fist), (1220, 690, d_fist)],
    "D": [(1320, 640, lerp(d_fist, d_grip, 0.25)), (1440, 610, lerp(d_fist, d_grip, 0.45)),
          (1560, 620, lerp(d_fist, d_grip, 0.65)), (1650, 680, lerp(d_fist, d_grip, 0.85)), (1690, 820, d_grip)],
    "E": [(1700, 1060, d_grip), (1760, 1230, lerp(d_grip, d_chest - 0.05, 0.6)), (1800, 1380, d_chest - 0.05),
          (1830, 1530, d_chest - 0.04), (1880, 1760, d_flank - 0.04), (1940, 1990, d_lap - 0.04)],
    "F": [(1500, 1060, d_rarm - 0.05), (1540, 1220, min(d_rarm, d_flank) - 0.05),
          (1580, 1400, min(d_rarm, d_flank) - 0.05), (1650, 1600, d_flank - 0.05), (1760, 1740, d_flank - 0.05)],
    "G": [(1760, 1990, d_lap - 0.06), (1820, 2120, d_lap - 0.04), (1870, 2260, d_lap - 0.02), (1800, 2370, d_lap - 0.01)],
    "H": [(1110, 1270, d_h), (1150, 1500, d_h), (1190, 1700, d_h), (1230, 1880, lerp(d_h, d_knee, 0.5)),
          (1320, 1990, d_knee - 0.01), (1440, 2010, d_knee - 0.02), (1580, 1950, d_lap - 0.03), (1700, 1880, d_lap - 0.04)],
}


def film_depth(mask, parts, S=8):
    """Depth along the strip: fixed at the points above, carried smoothly along
    the strip between them (stiff along it, loose across the gaps)."""
    h, w = H // S, W // S
    m = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.05
    vals = np.zeros((h, w), np.float32)
    fixed = np.zeros((h, w), bool)
    ys, xs = np.where(m)
    for part in parts:
        for (x, y, d) in FILM_POINTS[part]:
            gx, gy = x / S, y / S
            if len(xs):                                          # snap the point onto the strip
                k = np.argmin((xs - gx) ** 2 + (ys - gy) ** 2)
                if (xs[k] - gx) ** 2 + (ys[k] - gy) ** 2 < (60 / S) ** 2:
                    gx, gy = xs[k], ys[k]
            cv2.circle(vals, (int(gx), int(gy)), 2, float(d), -1)
            cv2.circle(fixed.view(np.uint8), (int(gx), int(gy)), 2, 1, -1)
    d = laplace_fill(vals, fixed, 1.0 + 300.0 * m)
    return cv2.GaussianBlur(cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), 6).astype(np.float32)


d_fb = film_depth(a_fb > 0.02, ["A", "B", "H"])
d_ff = film_depth(a_ff > 0.02, ["C", "D", "E", "F", "G", "H"])
# the front strands always hang in front of whatever part of him is behind them
d_ff = np.where(a_boy > 0.5, np.minimum(d_ff, d_boy - 0.015), d_ff).astype(np.float32)

# ============ colours: from the nearest layer back =============================
# Each layer's colour is solved so that, seen from the photo's viewpoint, all
# layers together give back the photo exactly. Going from the nearest layer
# back, a layer's soft edge keeps its own colour (the nearest solid part of it)
# and what lies behind it is read from the photo, held close to a painting of
# that place without the layer. Under the solid part of a layer only the
# painting is used.


def own_colour(alpha, thresh=0.95, grow=2):
    core = cv2.erode((alpha > thresh).astype(np.uint8), ell(grow)) > 0
    if not core.any():
        core = alpha > alpha.max() * 0.8
    f = nearest_fill(C, core)
    return np.where(core[..., None], C, cv2.GaussianBlur(f, (0, 0), 2.5)).astype(np.float32)


def peel(T, a, own, plate, lo=0.93, hi=1.10):
    a3 = a[..., None]
    read = (T - a3 * own) / np.maximum(1 - a3, 1e-3)
    read = np.clip(read, plate * lo - 4, plate * hi + 5)
    w = smoothstep(0.5, 0.9, a)[..., None]
    below = np.where(a3 > 0, read * (1 - w) + plate * w, T)
    P = np.clip(T - (1 - a3) * below, 0, 255)
    return P.astype(np.float32), below.astype(np.float32)


# --- paintings of what each layer hides ---
# under the film and the hand: him (his chest, arm, shorts) or the room
cover_front = cv2.dilate(((a_ff > 0.3) | (a_hand > 0.3)).astype(np.uint8), ell(6)) > 0
plate_ff = cached("plate_film_front", lambda: lama(np.clip(C, 0, 255).astype(np.uint8), cover_front, max_side=2048, pad=260))
# under his near arm and the strips on his chest the earlier, careful painting
# (collarbone, necklace cord, skin) is better: use it there
HIDDEN_SEED = 13
hidden = cached(f"boy_hidden_{HIDDEN_SEED}", lambda: C.copy())
old_front = cv2.imread(str(CACHE / "alpha_front_old.png"), 0)
hidden_ok = np.zeros((H, W), bool)
if old_front is not None:
    hidden_ok = (cv2.dilate((old_front > 5).astype(np.uint8), ell(6)) > 0)
hidden_ok &= (xx_ > 1280) & (xx_ < 2720) & (yy_ > 620) & (yy_ < 2140) & (body_sil > 0.5)
w_h = cv2.GaussianBlur(hidden_ok.astype(np.float32), (0, 0), 3)[..., None]
plate_ff = plate_ff * (1 - w_h) + hidden * w_h


def skin_grain(shape, seed):
    """Fine noise with the grain of the photo's own skin (a smooth fill alone looks airbrushed)."""
    patch = C[1420:1700, 1950:2300]
    hp = patch - cv2.GaussianBlur(patch, (0, 0), 2.0)
    sd = hp.reshape(-1, 3).std(0)
    n = np.random.default_rng(seed).standard_normal(shape[:2]).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.7)[..., None]
    n = n / max(float(n.std()), 1e-6)
    return n * sd[None, None, :]


# On his chest and belly, behind the strips, the skin is plain: a smooth
# continuation of the skin around them (with the skin's grain) is exact there
# and leaves no trace of a strip. Off his body, a smooth continuation of the
# dark gap beside him.
gx0, gy0, gx1, gy1 = 1380, 700, 2720, 2160
Kb = 2
sm = lambda x: cv2.resize(x[gy0:gy1, gx0:gx1].astype(np.float32), ((gx1 - gx0) // Kb, (gy1 - gy0) // Kb),
                          interpolation=cv2.INTER_AREA)
bg_ = lambda x: cv2.resize(x, (gx1 - gx0, gy1 - gy0), interpolation=cv2.INTER_LINEAR)
strips = cv2.dilate((a_ff > 0.05).astype(np.uint8), ell(10)) > 0
strips &= (xx_ > 1640) & (yy_ > 1040)                              # E and the lower F, not the hand area
# ...and where the earlier painting carried the brown strip on up under his
# arm towards the collarbone (a second copy of it, now that the strip moves)
up_band = np.zeros((H, W), np.uint8)
cv2.fillPoly(up_band, [np.array([(1735, 860), (1800, 860), (1822, 1050), (1745, 1050)], np.int32)], 1)
strips |= (up_band > 0) & hidden_ok
known_skin = (a_vit > 0.9) & ~strips & ~(cv2.dilate((cov_all > 0.05).astype(np.uint8), ell(10)) > 0)
known_gap = (a_vit < 0.3) & ~strips & ~(cv2.dilate((cov_all > 0.05).astype(np.uint8), ell(10)) > 0)
skin_lap = bg_(laplace_fill(sm(C), sm(known_skin) > 0.99))
gap_lap = bg_(laplace_fill(sm(C), sm(known_gap) > 0.99))
sil = body_sil[gy0:gy1, gx0:gx1, None]
target = sil * (skin_lap + skin_grain(skin_lap.shape, 4)) + (1 - sil) * gap_lap
wz = cv2.GaussianBlur(strips[gy0:gy1, gx0:gx1].astype(np.float32), (0, 0), 3)[..., None]
plate_ff[gy0:gy1, gx0:gx1] = plate_ff[gy0:gy1, gx0:gx1] * (1 - wz) + target * wz

# The forearm's dark underside rim, just outside the hand's cut-out, belongs
# to the arm (it must leave with it, not stay on his chest as a line): the
# hand layer carries it as a darkening, the skin it lies on is plain skin.
fx0, fy0, fx1, fy1 = 1760, 980, 2720, 1720
fa = a_hand[fy0:fy1, fx0:fx1]
near_h = cv2.dilate((fa > 0.05).astype(np.uint8), ell(18)) > 0
rim_b = near_h & (fa < 0.5) & (a_vit[fy0:fy1, fx0:fx1] > 0.9)
rim_b[:220] = False
rim_b[:, :320] = False
known_r = (a_vit[fy0:fy1, fx0:fx1] > 0.9) & ~(cv2.dilate((fa > 0.05).astype(np.uint8), ell(24)) > 0)
half_ = lambda x: cv2.resize(x.astype(np.float32), ((fx1 - fx0) // 2, (fy1 - fy0) // 2), interpolation=cv2.INTER_AREA)
skin_r = cv2.resize(laplace_fill(half_(C[fy0:fy1, fx0:fx1]), half_(known_r) > 0.99), (fx1 - fx0, fy1 - fy0),
                    interpolation=cv2.INTER_LINEAR) + skin_grain((fy1 - fy0, fx1 - fx0), 5)
wr = cv2.GaussianBlur(rim_b.astype(np.float32), (0, 0), 2)
a_rim = np.clip(1 - (lum(C[fy0:fy1, fx0:fx1]) + 1) / (lum(skin_r) + 1), 0, 0.9) * wr
a_hand[fy0:fy1, fx0:fx1] = np.maximum(fa, a_rim)
plate_ff[fy0:fy1, fx0:fx1] = plate_ff[fy0:fy1, fx0:fx1] * (1 - wr[..., None]) + skin_r * wr[..., None]

# under the boy: the room (wall continued behind him, the trunk and cans)
B_plate = cv2.imread(str(ROOT / "tools/voyager/room_clean.png")).astype(np.float32)
R_wall = np.where(veil_zone[..., None], B, FAR)                 # the wall behind the blurred film: the photo with it taken out
R_room = a_obj[..., None] * B_plate + (1 - a_obj[..., None]) * R_wall

# --- peel, nearest first ---
own_hand = own_colour(a_hand)
own_hand[fy0:fy1, fx0:fx1] *= (1 - (a_rim > fa)[..., None])        # the rim is a darkening: no colour of its own
core_h = cv2.dilate((a_hand > 0.5).astype(np.uint8), ell(3)) > 0
P_hand, T1 = peel(C, a_hand, own_hand, np.where(core_h[..., None], plate_ff, C))

own_ff = own_colour(a_ff, 0.9, 1)
core_f = cv2.dilate((a_ff > 0.5).astype(np.uint8), ell(3)) > 0
P_ff, T2 = peel(T1, a_ff, own_ff, np.where(core_f[..., None], plate_ff, T1))

own_boy = own_colour(a_boy * (1 - cov_all))
core_b = cv2.dilate((a_boy > 0.5).astype(np.uint8), ell(3)) > 0
P_boy, T3 = peel(T2, a_boy, own_boy, np.where(core_b[..., None], R_room, T2))

own_fb = np.where(ab_zone[..., None], film[None, None, :] * np.ones_like(C), own_colour(a_fb * (~ab_zone), 0.9, 1))
P_fb, T4 = peel(T3, a_fb, own_fb, R_room)

# the room: trunk and cans over the wall and floor. Soft edge pixels of the
# cans that look just like the wall behind them are wall (else the cans carry
# a pale halo when they move).
same_o = cv2.GaussianBlur(np.abs(T4 - FAR).max(2), (0, 0), 1.5)
a_obj = (a_obj * np.maximum(smoothstep(5, 14, same_o), smoothstep(0.85, 0.97, a_obj))).astype(np.float32)
a_obj[a_obj < 0.012] = 0
a_obj_ext[PAD_T:PAD_T + H, PAD_L:PAD_L + W] = a_obj
core_o = cv2.erode((a_obj > 0.95).astype(np.uint8), ell(2)) > 0
own_obj = np.where(core_o[..., None], B_plate, cv2.GaussianBlur(nearest_fill(B_plate, core_o), (0, 0), 2.5))
P_obj, S0 = peel(T4, a_obj, own_obj, FAR)
S1 = P_obj + (1 - a_obj[..., None]) * S0
S2 = P_boy + (1 - a_boy[..., None]) * (P_fb + (1 - a_fb[..., None]) * S1)
S4 = P_hand + (1 - a_hand[..., None]) * (P_ff + (1 - a_ff[..., None]) * S2)
err = np.abs(S4 - C)

print("home-view error: mean %.3f, 99.9th pct %.1f, max %.1f" % (err.mean(), np.percentile(err, 99.9), err.max()))

# ============ write ============================================================
layers = []


def bbox(mask, margin=24):
    ys, xs = np.where(mask)
    x0, x1 = max(0, xs.min() - margin), min(mask.shape[1], xs.max() + margin + 1)
    y0, y1 = max(0, ys.min() - margin), min(mask.shape[0], ys.max() + margin + 1)
    x0, y0 = x0 // STEP * STEP, y0 // STEP * STEP
    return x0, y0, x1, y1


def write_picture(name, P, a):
    """Colour as WebP (high quality) and transparency as a separate lossless
    PNG, full and half size. They are kept apart because the colour is
    premultiplied: it must stay exactly black where the layer is transparent,
    which a lossy image with its own alpha channel does not guarantee."""
    q = [cv2.IMWRITE_WEBP_QUALITY, 96]
    rgb = np.round(np.clip(P, 0, 255)).astype(np.uint8)
    al = None if a is None else np.round(a * 255).astype(np.uint8)
    if al is not None:
        rgb[al == 0] = 0
    for tag, k in (("", 1), ("@half", 2)):
        size = (max(1, rgb.shape[1] // k), max(1, rgb.shape[0] // k))
        cv2.imwrite(str(OUT / f"{name}{tag}.webp"), rgb if k == 1 else cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), q)
        if al is not None:
            cv2.imwrite(str(OUT / f"{name}{tag}-alpha.png"), al if k == 1 else cv2.resize(al, size, interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])


def add_layer(name, P, a, d, origin=(0, 0), step=STEP, scale=1.0, crop=True, opaque=False, guard=(0, 0)):
    """P, a, d are arrays on a canvas whose top-left is `origin` in photo pixels, at `scale` px per photo px."""
    if crop and a is not None:
        x0, y0, x1, y1 = bbox(a > 0)
    else:
        x0, y0, x1, y1 = 0, 0, P.shape[1], P.shape[0]
    P, d = P[y0:y1, x0:x1], d[y0:y1, x0:x1]
    write_picture(name, P, None if a is None else a[y0:y1, x0:x1])
    # depth sampled at the corners of step-sized cells, edge to edge, so the mesh covers the whole picture
    h, w = d.shape
    gw, gh = -(-w // step) + 1, -(-h // step) + 1
    ds = cv2.GaussianBlur(d, (0, 0), step / 3)
    ys = np.minimum(np.arange(gh) * step, h - 1)
    xs = np.minimum(np.arange(gw) * step, w - 1)
    g = ds[np.ix_(ys, xs)]
    inv = 1.0 / g
    lo, hi = float(inv.min()), float(inv.max())
    q = np.round((inv - lo) / max(hi - lo, 1e-9) * 65535).astype(np.uint16)
    img = np.zeros(q.shape + (3,), np.uint8)
    img[..., 2] = q >> 8
    img[..., 1] = q & 255
    cv2.imwrite(str(OUT / f"{name}-depth.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    layer = {
        "name": name, "color": f"{name}.webp", "color_half": f"{name}@half.webp", "depth": f"{name}-depth.png",
        "opaque": opaque, "guard": list(guard),
        "inv_min": lo, "inv_max": hi, "grid": [gw, gh],
        # photo-pixel rectangle the layer's image covers, and photo pixels per depth-map cell
        "rect": [origin[0] + x0 / scale, origin[1] + y0 / scale, w / scale, h / scale],
        "cell": step / scale,
    }
    if a is not None:
        layer["alpha"], layer["alpha_half"] = f"{name}-alpha.png", f"{name}@half-alpha.png"
    layers.append(layer)
    print(name, w, "x", h, "grid", gw, gh)


# surround: half resolution, whole extended canvas
d_half = cv2.resize(d_far_ext, (EW // 2, EH // 2), interpolation=cv2.INTER_AREA)
add_layer("surround", far_half_img, None, d_half, origin=(-PAD_L, -PAD_T), step=3, scale=0.5, crop=False, opaque=True)

# far: the frame itself, full resolution, edges feathered into the surround
# The page always crops the photo a little (zoom 1.06), so its outermost strip
# is never on screen at rest. Within that strip the photo fades into the
# painted room around it, so the join is a gradual blend, not a straight line,
# when the camera moves. (Widths keep inside the strip the page hides: about
# 3% at the sides and bottom, 2% at the top with the crop anchored 40% down.)
xs_ = np.arange(W, dtype=np.float32)[None, :]
ys_ = np.arange(H, dtype=np.float32)[:, None]
edge = (smoothstep(0, 90, xs_) * smoothstep(0, 90, W - 1 - xs_) *
        smoothstep(0, 50, ys_) * smoothstep(0, 75, H - 1 - ys_)).astype(np.float32)
add_layer("far", S0 * edge[..., None], edge, d_far, crop=False)

# objects: full resolution in the frame, continued into the margin from the half-resolution room
room_up = cv2.resize(room_half, (EW, EH), interpolation=cv2.INTER_CUBIC)
far_up = cv2.resize(far_half_img, (EW, EH), interpolation=cv2.INTER_CUBIC)
P_obj_ext = np.clip(room_up - (1 - a_obj_ext[..., None]) * far_up, 0, 255)
# inside the frame the full-resolution trunk and cans; within the strip the
# page always crops off, they blend into the painted continuation, so there is
# no seam at the photo's edge when the camera moves
wl = smoothstep(0, 90, np.arange(W, dtype=np.float32))[None, :, None]
P_obj_ext[PAD_T:PAD_T + H, PAD_L:PAD_L + W] = P_obj * wl + P_obj_ext[PAD_T:PAD_T + H, PAD_L:PAD_L + W] * (1 - wl)
add_layer("objects", P_obj_ext, a_obj_ext, d_obj_ext, origin=(-PAD_L, -PAD_T))

add_layer("film_back", P_fb, a_fb, d_fb)
add_layer("boy", P_boy, a_boy, d_boy)
add_layer("film_front", P_ff, a_ff, d_ff)
add_layer("hand", P_hand, a_hand, d_hand)

scene = {
    "width": W, "height": H, "focal": focal, "cx": W / 2, "cy": H / 2,
    "pivot_depth": float(np.median(dp[fg_solid])),
    "pad": [PAD_L, PAD_T, PAD_R, PAD_B],
    "layers": layers,
}
json.dump(scene, open(OUT / "scene.json", "w"), indent=1)

# previews for checking
cv2.imwrite(str(WORK / "prev_film.png"), (np.dstack([a_fb, a_ff, a_hand]) * 255).astype(np.uint8)[::3, ::3])
cv2.imwrite(str(WORK / "prev_back.jpg"), np.clip(S2, 0, 255).astype(np.uint8)[::2, ::2])
cv2.imwrite(str(WORK / "prev_room_ext.jpg"), np.clip(room_half, 0, 255).astype(np.uint8)[::2, ::2])
cv2.imwrite(str(WORK / "prev_far_ext.jpg"), np.clip(far_half_img, 0, 255).astype(np.uint8)[::2, ::2])
