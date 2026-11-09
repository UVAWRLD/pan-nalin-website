# Builds Voyager's condition inputs for the storeroom, one folder per camera path.
#
# Unlike Voyager's own create_input.py (fixed 16:9 frame, fx=256, 1-unit
# moves), this uses the photo's real camera, solved in Blender, at Voyager's
# native 3:2 size, and short moves sized to the room, so the generated views
# stay close to where the site's camera actually goes.
#
# Run from HunyuanWorld-Voyager/data_engine:
#   python make_inputs.py --image room_clean.png --out ../inputs
# Writes ../inputs/<path>/ (ref_image.png, ref_depth.exr, video_input/,
# depth_range.json, cameras.npz) for every path in PATHS.

import argparse, json, os
import numpy as np
import torch
from PIL import Image
from create_input import (depth_to_world_coords_points, render_from_cameras_videos,
                          create_video_input)
try:
    from moge.model.v1 import MoGeModel
except ImportError:
    from MoGe.moge.model.v1 import MoGeModel

# The photo is 3840x2731. Crop 85 px top and bottom to 3:2, then work at
# 1152x768 (Voyager resizes to 768x512 itself).
PHOTO_W, PHOTO_H, CROP_TOP = 3840, 2731, 85
W, H = 1152, 768
S = W / PHOTO_W
VFOV = 29.06                       # photo camera, vertical, degrees
SHIFT = (34.35, 80.5)              # optical centre offset in photo px
FY = (PHOTO_H / 2) / np.tan(np.radians(VFOV) / 2) * S
CX = (PHOTO_W / 2 + SHIFT[0]) * S
CY = (PHOTO_H / 2 + SHIFT[1] - CROP_TOP) * S
FRAMES = 49

# Camera paths in OpenCV axes (x right, y down, z forward), as fractions of
# the distance D to the middle of the room. Each path looks at the point D
# ahead of the start, so the camera orbits the space instead of drifting.
PATHS = {
    "orbit_left":  np.array([-0.20, 0.00, 0.05]),
    "orbit_right": np.array([0.20, 0.00, 0.05]),
    "push_in":     np.array([0.00, 0.00, 0.30]),
    "rise_left":   np.array([-0.10, -0.12, 0.10]),
}


def intrinsic(scale=1.0):
    return np.array([[FY * scale, 0, CX * scale], [0, FY * scale, CY * scale], [0, 0, 1]])


def look_at_w2c(eye, target):
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, -1.0, 0.0])); x /= np.linalg.norm(x)   # world up is -y
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = -R @ eye
    return w2c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="../inputs")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    assert img.size == (PHOTO_W, PHOTO_H), img.size
    img = img.crop((0, CROP_TOP, PHOTO_W, PHOTO_H - CROP_TOP)).resize((W, H), Image.LANCZOS)
    image = np.array(img)

    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").cuda()
    out = model.infer(torch.tensor(image / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1))
    depth = out["depth"].cpu().numpy()
    finite = np.isfinite(depth)
    depth[~finite] = depth[finite].max() + 1e4
    D = float(np.median(depth[H // 3: 2 * H // 3, W // 3: 2 * W // 3]))
    print("distance to the middle of the room:", D)

    identity = np.eye(4)
    points = depth_to_world_coords_points(depth, identity, intrinsic()).reshape(-1, 3)
    colors = image.reshape(-1, 3)
    target = np.array([0.0, 0.0, D])

    for name, end in PATHS.items():
        t = np.linspace(0, 1, FRAMES)
        ease = 0.5 - 0.5 * np.cos(np.pi * t)
        eyes = ease[:, None] * (end * D)[None]
        extr = np.stack([look_at_w2c(e, target) for e in eyes])
        intr = np.stack([intrinsic(0.5)] * FRAMES)
        renders, masks, depths = render_from_cameras_videos(
            points, colors, extr, intr, height=H // 2, width=W // 2)
        folder = os.path.join(args.out, name)
        create_video_input(renders, masks, depths, folder, separate=True,
                           ref_image=image, ref_depth=depth, Width=W, Height=H)
        np.savez(os.path.join(folder, "cameras.npz"), w2c=extr, K=intrinsic(), size=(W, H),
                 room_distance=D)
        print("wrote", folder)

    json.dump({"room_distance": D, "fy": FY, "cx": CX, "cy": CY, "size": [W, H]},
              open(os.path.join(args.out, "camera.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
