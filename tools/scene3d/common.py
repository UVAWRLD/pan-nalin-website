# Shared helpers for the scene builder.
import numpy as np
import cv2
import torch
from scipy.ndimage import distance_transform_edt


def ell(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def nearest_fill(arr, valid):
    """Every pixel takes the value of the nearest valid pixel."""
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return arr[idx[0], idx[1]]


def smooth_fill(arr, valid, levels=12):
    """Fill the invalid pixels with a smooth (push-pull) continuation of the valid ones."""
    arr = arr.astype(np.float32)
    w = valid.astype(np.float32)
    if arr.ndim == 3:
        w3 = w[..., None]
    else:
        w3 = w
    pyr = [(arr * w3, w)]
    for _ in range(levels):
        a, ww = pyr[-1]
        if min(ww.shape) < 4:
            break
        a2 = cv2.resize(a, (ww.shape[1] // 2, ww.shape[0] // 2), interpolation=cv2.INTER_AREA)
        w2 = cv2.resize(ww, (ww.shape[1] // 2, ww.shape[0] // 2), interpolation=cv2.INTER_AREA)
        pyr.append((a2, w2))
    a, ww = pyr[-1]
    cur = a / np.maximum(ww if a.ndim == 2 else ww[..., None], 1e-6)
    mean = a.reshape(-1, *a.shape[2:]).sum(0) / max(float(ww.sum()), 1e-6)   # for cells with nothing valid in reach
    cur[ww < 1e-4] = mean
    for a, ww in reversed(pyr[:-1]):
        up = cv2.resize(cur, (ww.shape[1], ww.shape[0]), interpolation=cv2.INTER_LINEAR)
        wn = np.clip(ww, 0, 1)
        if a.ndim == 3:
            own = a / np.maximum(ww[..., None], 1e-6)
            cur = own * wn[..., None] + up * (1 - wn[..., None])
        else:
            own = a / np.maximum(ww, 1e-6)
            cur = own * wn + up * (1 - wn)
    out = arr.copy()
    out[~valid] = cur[~valid]
    return out


_lama = None


def lama(image_bgr, mask, max_side=1536, pad=300, feather=3):
    """Inpaint mask (bool) with big-LaMa on a crop around it. Returns a new image."""
    global _lama
    if not mask.any():
        return image_bgr.copy()
    if _lama is None:
        from huggingface_hub import hf_hub_download
        _lama = torch.jit.load(hf_hub_download("fashn-ai/LaMa", "big-lama.pt"), map_location="cpu").eval()
    H, W = mask.shape
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    crop = image_bgr[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1].astype(np.uint8)
    h, w = m.shape
    s = min(1.0, max_side / max(h, w))
    hw = (int(round(h * s / 8)) * 8, int(round(w * s / 8)) * 8)
    c = cv2.resize(crop, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    mm = (cv2.resize(m * 255, (hw[1], hw[0]), interpolation=cv2.INTER_AREA) > 0).astype(np.float32)
    mm = cv2.dilate(mm, ell(2))
    img_t = torch.from_numpy(c[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255
    m_t = torch.from_numpy(mm)[None, None]
    with torch.no_grad():
        out = _lama(img_t, m_t)[0].permute(1, 2, 0).numpy()
    out = np.clip(out * 255, 0, 255)[:, :, ::-1]
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    a = cv2.GaussianBlur(cv2.dilate(m, ell(feather)).astype(np.float32), (0, 0), feather)[..., None]
    a = np.maximum(a, m[..., None].astype(np.float32))
    res = image_bgr.copy()
    res[y0:y1, x0:x1] = np.clip(out * a + crop.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)
    return res


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
