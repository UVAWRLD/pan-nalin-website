# First half of the scene build: everything build.py needs that comes from
# running models on the photo. Slow (about 10 minutes), and only needs running
# again if the photo or the empty-room plate changes. Writes into work/.
#
#   python tools/scene3d/prepare.py
#
#   work/depth_photo.npz         metric depth of the photo, and the focal length (Depth Pro)
#   work/depth_room.npz          the same for the empty-room plate
#   work/depth_room_aligned.npy  the plate's depth brought to the photo's scale
#   work/alpha_fg.png            the boy and his film, cut out to the hair (ViTMatte)
#   work/alpha_obj.png           the trunk and the film cans, cut out of the plate
import json, subprocess, sys
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ell

ROOT = HERE.parents[1]
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)
PHOTO = ROOT / "assets/img/hero-boy-film-strip.png"
PLATE = ROOT / "tools/voyager/room_clean.png"     # the room with the boy painted out
PY = sys.executable


def run(*args):
    subprocess.run([PY, *map(str, args)], check=True)


def step(path, fn):
    if Path(path).exists():
        print("have", Path(path).name)
    else:
        print("making", Path(path).name, flush=True)
        fn()


# ---- depth --------------------------------------------------------------------------
step(WORK / "depth_photo.npz", lambda: run(HERE / "depth.py", PHOTO, WORK / "depth_photo.npz"))
step(WORK / "depth_room.npz", lambda: run(HERE / "depth.py", PLATE, WORK / "depth_room.npz"))

C = cv2.imread(str(PHOTO))
B = cv2.imread(str(PLATE))
dp = np.load(WORK / "depth_photo.npz")["depth"]
diff_s = cv2.GaussianBlur(np.abs(C.astype(np.int16) - B.astype(np.int16)).max(2).astype(np.float32), (0, 0), 2)


def align():
    # Depth Pro guesses scale separately for each image: match the plate's inverse
    # depth to the photo's (a, b in 1/d' = a/d + b) where the two images agree.
    dr = np.load(WORK / "depth_room.npz")["depth"]
    same = cv2.dilate((diff_s > 3).astype(np.uint8), np.ones((61, 61), np.uint8)) == 0
    x, y = 1 / dr[same], 1 / dp[same]
    A = np.stack([x, np.ones_like(x)], 1)
    a, b = np.linalg.lstsq(A[::50], y[::50], rcond=None)[0]
    np.save(WORK / "depth_room_aligned.npy", (1 / (a / dr + b)).astype(np.float32))


step(WORK / "depth_room_aligned.npy", align)
dra = np.load(WORK / "depth_room_aligned.npy")


# ---- the boy and his film --------------------------------------------------------
def boy_matte():
    # Sure foreground: well in front of the empty room and near the camera, or very
    # different from the plate. Unsure: anything that differs from the plate at all.
    gap = dra - dp
    changed = cv2.morphologyEx((diff_s > 4).astype(np.uint8), cv2.MORPH_OPEN, ell(3))
    maybe = cv2.dilate(changed, ell(14))
    solid = (((gap > 0.25) & (dp < 2.35)) | (diff_s > 60)).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, ell(4))
    tri = np.zeros(dp.shape, np.uint8)
    tri[maybe > 0] = 128
    tri[cv2.erode(solid, ell(9)) > 0] = 255
    cv2.imwrite(str(WORK / "trimap_fg.png"), tri)
    run(HERE / "matte.py", PHOTO, WORK / "trimap_fg.png", WORK / "alpha_fg.png")


step(WORK / "alpha_fg.png", boy_matte)


# ---- the trunk and the film cans -------------------------------------------------
def objects_matte():
    run(HERE / "segment.py", PLATE, HERE / "prompts_room.json", WORK / "seg_room")
    obj = (cv2.imread(str(WORK / "seg_room/cans.png"), 0) > 0) | (cv2.imread(str(WORK / "seg_room/trunk.png"), 0) > 0)
    obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_CLOSE, ell(25))
    h, w = obj.shape                                    # fill holes
    inv = (1 - obj).astype(np.uint8)
    cv2.floodFill(inv, np.zeros((h + 2, w + 2), np.uint8), (w - 1, 0), 0)
    obj = (obj | inv).astype(np.uint8)
    tri = np.zeros(obj.shape, np.uint8)
    tri[cv2.dilate(obj, ell(16)) > 0] = 128
    tri[cv2.erode(obj, ell(16)) > 0] = 255
    cv2.imwrite(str(WORK / "trimap_obj.png"), tri)
    run(HERE / "matte.py", PLATE, WORK / "trimap_obj.png", WORK / "alpha_obj.png")


step(WORK / "alpha_obj.png", objects_matte)
print("prepared; now run tools/scene3d/build.py")
