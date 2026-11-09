# Turns Voyager's RGB-D videos into one Gaussian splat of the storeroom.
#
# Every video shares the same world (the photo camera at the origin, OpenCV
# axes), so the frames, their known cameras and their depth go straight into
# one optimisation: points from the depth start the Gaussians, then gsplat
# fits them to all frames plus the full-size room image.
#
# Run from HunyuanWorld-Voyager:
#   python to_splat.py --inputs inputs --results results --image room_clean.png --out room.ply

import argparse, glob, json, math, os
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

C0 = 0.28209479177387814


def load_views(inputs, results):
    views = []
    for folder in sorted(glob.glob(os.path.join(inputs, "*/cameras.npz"))):
        name = os.path.basename(os.path.dirname(folder))
        videos = sorted(glob.glob(os.path.join(results, name, "*.mp4")))
        if not videos:
            print("no video for", name); continue
        cams = np.load(folder)
        ranges = json.load(open(os.path.join(os.path.dirname(folder), "depth_range.json")))
        frames = [f for f in imageio.get_reader(videos[-1])]
        h = frames[0].shape[0] // 2
        w = frames[0].shape[1]
        W0, H0 = cams["size"]
        K = cams["K"].copy(); K[0] *= w / W0; K[1] *= h / H0
        for i, f in enumerate(frames[: len(cams["w2c"])]):
            rgb = f[:h].astype(np.float32) / 255
            inv = f[h:, :, 0].astype(np.float32) / 255
            r0, r1 = ranges[i]
            depth = 1 / (inv * (r1 - r0) + r0 + 1e-6)
            views.append(dict(rgb=rgb, depth=depth, K=K, w2c=cams["w2c"][i], name=f"{name}/{i}"))
        print(name, len(frames), "frames", (w, h))
    return views, float(cams["room_distance"])


def backproject(v, stride, max_depth):
    h, w = v["depth"].shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    d = v["depth"][ys, xs]
    ok = (d > 0) & (d < max_depth)
    K = v["K"]
    pc = np.stack([(xs - K[0, 2]) * d / K[0, 0], (ys - K[1, 2]) * d / K[1, 1], d], -1)[ok]
    c2w = np.linalg.inv(v["w2c"])
    return pc @ c2w[:3, :3].T + c2w[:3, 3], v["rgb"][ys, xs][ok]


def ssim(a, b):
    k = torch.tensor([math.exp(-(x - 5) ** 2 / 4.5) for x in range(11)], device=a.device)
    k = (k / k.sum())[:, None] @ (k / k.sum())[None]
    k = k.expand(3, 1, 11, 11)
    f = lambda x: F.conv2d(x, k, padding=5, groups=3)
    mu_a, mu_b = f(a), f(b)
    va, vb, cov = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    return (((2 * mu_a * mu_b + 1e-4) * (2 * cov + 9e-4)) /
            ((mu_a ** 2 + mu_b ** 2 + 1e-4) * (va + vb + 9e-4))).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="inputs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--image", default="room_clean.png")
    ap.add_argument("--out", default="room.ply")
    ap.add_argument("--steps", type=int, default=12000)
    args = ap.parse_args()
    dev = "cuda"

    views, D = load_views(args.inputs, args.results)
    cam = json.load(open(os.path.join(args.inputs, "camera.json")))
    # The full-size room image is the sharpest view: add it, weighted up.
    W, H = cam["size"]
    img = Image.open(args.image).convert("RGB")
    crop = (img.height - H * img.width / W) / 2
    img = img.crop((0, crop, img.width, img.height - crop)).resize((W, H), Image.LANCZOS)
    K = np.array([[cam["fy"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]])
    photo = dict(rgb=np.asarray(img, np.float32) / 255, K=K, w2c=np.eye(4), name="photo", weight=4.0)

    pts, cols = zip(*[backproject(v, 4, 4 * D) for v in views[::3]])
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    keep = np.random.default_rng(0).choice(len(pts), min(len(pts), 600_000), replace=False)
    pts, cols = pts[keep], cols[keep]
    dist = cKDTree(pts).query(pts, k=4)[0][:, 1:].mean(1)
    print("init gaussians:", len(pts))

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(pts, dtype=torch.float32)),
        "scales": torch.nn.Parameter(torch.log(torch.tensor(dist, dtype=torch.float32).clamp(1e-4))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(len(pts), 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((len(pts),), 0.5))),
        "sh0": torch.nn.Parameter(((torch.tensor(cols, dtype=torch.float32) - 0.5) / C0)[:, None]),
    }).to(dev)
    lrs = {"means": 1.6e-4 * D, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3}
    opt = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    strategy = DefaultStrategy(refine_stop_iter=int(args.steps * 0.6), verbose=True)
    strategy.check_sanity(params, opt)
    state = strategy.initialize_state(scene_scale=D)
    mean_decay = 0.01 ** (1 / args.steps)

    train = [dict(v, weight=1.0) for v in views] + [photo] * 8
    gpu = [dict(rgb=torch.tensor(v["rgb"], device=dev), K=torch.tensor(v["K"], dtype=torch.float32, device=dev),
                w2c=torch.tensor(v["w2c"], dtype=torch.float32, device=dev), weight=v["weight"]) for v in train]
    rng = np.random.default_rng(1)
    for step in range(args.steps):
        v = gpu[rng.integers(len(gpu))]
        h, w = v["rgb"].shape[:2]
        out, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), params["sh0"], v["w2c"][None], v["K"][None],
            w, h, sh_degree=0, packed=False, absgrad=False)
        pred = out[0].clamp(0, 1)
        l1 = (pred - v["rgb"]).abs().mean()
        a, b = pred.permute(2, 0, 1)[None], v["rgb"].permute(2, 0, 1)[None]
        loss = v["weight"] * (0.8 * l1 + 0.2 * (1 - ssim(a, b)))
        strategy.step_pre_backward(params, opt, state, step, info)
        loss.backward()
        for o in opt.values():
            o.step(); o.zero_grad(set_to_none=True)
        for g in opt["means"].param_groups:
            g["lr"] *= mean_decay
        strategy.step_post_backward(params, opt, state, step, info, packed=False)
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f} gaussians {len(params['means'])}")

    # Standard 3DGS .ply (Spark reads it; convert to .spz for the web).
    p = {k: v.detach().cpu().numpy() for k, v in params.items()}
    n = len(p["means"])
    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    quats = p["quats"] / np.linalg.norm(p["quats"], axis=1, keepdims=True)
    data = np.concatenate([p["means"], np.zeros((n, 3)), p["sh0"][:, 0], p["opacities"][:, None],
                           p["scales"], quats], 1).astype(np.float32)
    el = np.empty(n, dtype=[(f, "f4") for f in fields])
    for i, f in enumerate(fields):
        el[f] = data[:, i]
    PlyData([PlyElement.describe(el, "vertex")]).write(args.out)
    print("wrote", args.out, n, "gaussians")


if __name__ == "__main__":
    main()
