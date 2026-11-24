# Continues a picture beyond its frame with a diffusion inpainting model
# (Stable Diffusion 1.5 inpainting, run locally), in overlapping 512-pixel
# tiles that each see plenty of the real picture.
import numpy as np
import cv2
import torch

PROMPT = ("dim interior of an old room, weathered plaster wall with peeling paint, rough stone floor, "
          "warm low light from the side, deep shadows, shallow depth of field, soft out of focus background, "
          "film still, photograph")
NEGATIVE = ("person, people, child, face, hands, text, letters, writing, sign, poster, window, door, lamp, "
            "furniture, bright, overexposed, frame, border, vignette, watermark, illustration, painting")


def outpaint(canvas_bgr, known, laplace_fill, init=None, strength=1.0, band=0, steps=32, guidance=6.0, seed=7, log=print,
             prompt=None, negative=None):
    """canvas_bgr: uint8 image whose `known` (bool) pixels are real; the rest is painted.
    Sized for 512-pixel tiles (work at about a quarter of the photo's resolution).
    init: a full-size first guess for the unknown part (e.g. a plain inpainting);
    with strength < 1 the model only refines it, keeping its tone and layout.
    band: also repaint this many pixels of the real picture along the border,
    blended, so the join is not a straight line."""
    from diffusers import StableDiffusionInpaintPipeline
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting", variant="fp16", torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    from PIL import Image

    H, W = known.shape
    img = canvas_bgr.astype(np.float32).copy()
    real = known.copy()
    if band:
        known = cv2.erode(known.astype(np.uint8), np.ones((2 * band + 1, 2 * band + 1), np.uint8)) > 0
    T = 512

    def starts(n):
        if n <= T:
            return [0]
        k = int(np.ceil((n - T) / (T * 0.6))) + 1
        return sorted({int(round(i * (n - T) / (k - 1))) for i in range(k)})

    # tiles along the border first (they hold the most real picture), any order is fine after that
    tiles = [(x, y) for y in starts(H) for x in starts(W)]
    tiles.sort(key=lambda t: -known[t[1]:t[1] + T, t[0]:t[0] + T].mean())
    g = torch.Generator(device="cpu").manual_seed(seed)
    for n, (x0, y0) in enumerate(tiles):
        k = known[y0:y0 + T, x0:x0 + T]
        if k.all():
            continue
        tile = img[y0:y0 + T, x0:x0 + T]
        if init is not None:
            guess = init[y0:y0 + T, x0:x0 + T].astype(np.float32)
        else:   # a smooth continuation of the known part
            guess = laplace_fill(cv2.resize(tile, (128, 128), interpolation=cv2.INTER_AREA),
                                 cv2.resize(k.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA) > 0.99)
            guess = cv2.resize(guess, (T, T), interpolation=cv2.INTER_LINEAR)
        start = np.where(k[..., None], tile, guess)
        mask = cv2.dilate((~k).astype(np.uint8), np.ones((9, 9), np.uint8))
        out = pipe(prompt=prompt or PROMPT, negative_prompt=negative or NEGATIVE,
                   image=Image.fromarray(np.clip(start, 0, 255).astype(np.uint8)[:, :, ::-1]),
                   mask_image=Image.fromarray(mask * 255), num_inference_steps=steps, guidance_scale=guidance,
                   strength=strength, generator=g).images[0]
        out = np.asarray(out)[:, :, ::-1].astype(np.float32)
        # The model shifts tone slightly even where it was told to keep the picture.
        # Measure that shift on the real pixels, continue it smoothly into the painted
        # part and take it out, so the painted part meets the photo without a step.
        shift = out - tile
        s = cv2.resize(np.where(k[..., None], shift, 0), (128, 128), interpolation=cv2.INTER_AREA)
        kw = cv2.resize(k.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        s = s / np.maximum(kw[..., None], 1e-3)
        s = laplace_fill(cv2.GaussianBlur(s, (0, 0), 2), kw > 0.5)
        out = out - cv2.resize(s, (T, T), interpolation=cv2.INTER_LINEAR)
        # inside the real picture the band blends gradually from painted (at the edge) to real
        a = cv2.GaussianBlur((~k).astype(np.float32), (0, 0), 1.5)[..., None]
        a = np.maximum(a, (~k)[..., None].astype(np.float32))
        if band:
            r = real[y0:y0 + T, x0:x0 + T]
            dist = cv2.distanceTransform(r.astype(np.uint8), cv2.DIST_L2, 5)
            a = np.where(r[..., None], np.clip(1 - dist / band, 0, 1)[..., None] * a, a)
        img[y0:y0 + T, x0:x0 + T] = np.where(k[..., None], tile, out * a + tile * (1 - a))
        known[y0:y0 + T, x0:x0 + T] = True
        log(f"outpaint tile {n + 1}/{len(tiles)} at {x0},{y0}")
    del pipe
    return np.clip(img, 0, 255).astype(np.uint8)


HIDDEN_PROMPT = ("photograph, film still, close-up of a shirtless boy sitting, bare chest, neck and shoulder, "
                 "smooth warm brown skin, soft warm side light from the left, deep shadow, dark background, "
                 "shallow depth of field")
HIDDEN_NEGATIVE = ("hand, hands, fingers, arm, forearm, extra limbs, nipple, deformed, text, watermark, painting, "
                   "illustration, cartoon")


def paint_hidden(photo_bgr, init_bgr, mask, box, strength=0.65, seed=1, scale=0.5, steps=40, guidance=6.0):
    """Repaint what something in front hid (mask, bool) inside box = (x0, y0, x1, y1),
    starting from init_bgr (a plain inpainting, for tone and layout) and letting a
    diffusion model give it believable form: collarbone, cord, skin. Returns a
    full-size image equal to the photo outside the mask."""
    from diffusers import StableDiffusionInpaintPipeline
    from PIL import Image
    x0, y0, x1, y1 = box
    crop = photo_bgr[y0:y1, x0:x1].astype(np.float32)
    m = mask[y0:y1, x0:x1]
    w, h = int((x1 - x0) * scale) // 8 * 8, int((y1 - y0) * scale) // 8 * 8
    im = cv2.resize(init_bgr[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_AREA)
    mm = cv2.dilate(cv2.resize(m.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST), np.ones((7, 7), np.uint8))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting", variant="fp16", torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    out = pipe(prompt=HIDDEN_PROMPT, negative_prompt=HIDDEN_NEGATIVE, image=Image.fromarray(im[:, :, ::-1]),
               mask_image=Image.fromarray(mm), height=h, width=w, num_inference_steps=steps, guidance_scale=guidance,
               strength=strength, generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    del pipe
    out = cv2.resize(np.asarray(out)[:, :, ::-1].astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    # take out the slow tone drift the model adds, measured where the photo is known
    k = ~(cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    num = cv2.GaussianBlur(np.where(k[..., None], out - crop, 0), (0, 0), 40)
    den = cv2.GaussianBlur(k.astype(np.float32), (0, 0), 40)[..., None]
    out = out - num / np.maximum(den, 1e-3)
    a = np.maximum(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2), m.astype(np.float32))[..., None]
    res = photo_bgr.astype(np.float32).copy()
    res[y0:y1, x0:x1] = out * a + crop * (1 - a)
    return np.clip(res, 0, 255).astype(np.uint8)
