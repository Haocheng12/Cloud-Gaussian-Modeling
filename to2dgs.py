#!/usr/bin/env python3
"""
Cloud-only 2D Gaussian splatting (with depth "z").

- Segments clouds in HSV (bright + low saturation).
- remove_edge_touching_blobs(): drops connected components that touch the image border
  (optionally trims a small margin) so we don't sample incomplete edge clouds.
- Edge-aware, anisotropic covariances via a structure tensor.
- Adds a per-Gaussian depth z, used to:
    * sort: draw far → near (simple occlusion),
    * scale size by (z_ref / z): farther blobs appear smaller,
    * optional depth fog (fade with distance).
- Renders *only the cloud* (no sky) to:
    - cloud_only_black.png        (black background)
    - cloud_only_transparent.png  (RGBA with alpha)
- Saves model.json with {x,y,z,cov,amplitude,color,alpha}.

pip install numpy pillow
"""

import argparse, json, os
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from PIL import Image

# --------------------------- utils
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def to_uint8(img01: np.ndarray) -> np.ndarray:
    return (np.clip(img01, 0, 1) * 255.0 + 0.5).astype(np.uint8)

def rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    r,g,b = arr[...,0], arr[...,1], arr[...,2]
    mx, mn = arr.max(-1), arr.min(-1)
    v = mx
    s = (mx - mn) / (mx + 1e-8)
    rc = (mx - r) / (mx - mn + 1e-8)
    gc = (mx - g) / (mx - mn + 1e-8)
    bc = (mx - b) / (mx - mn + 1e-8)
    h = np.zeros_like(mx)
    h = np.where(mx==mn, 0.0, h)
    h = np.where((mx==r) & (mx!=mn), (bc - gc), h)
    h = np.where((mx==g) & (mx!=mn), 2.0 + (rc - bc), h)
    h = np.where((mx==b) & (mx!=mn), 4.0 + (gc - rc), h)
    h = (h/6.0) % 1.0
    return np.stack([h,s,v], -1).astype(np.float32)

def box_blur(img: np.ndarray, iters=1) -> np.ndarray:
    out = img.astype(np.float32)
    for _ in range(iters):
        p = np.pad(out, ((1,1),(1,1)), mode="reflect")
        out = (p[0:-2,0:-2]+p[0:-2,1:-1]+p[0:-2,2:]
             + p[1:-1,0:-2]+p[1:-1,1:-1]+p[1:-1,2:]
             + p[2:,0:-2]+p[2:,1:-1]+p[2:,2:]) / 9.0
    return out

def conv2(im: np.ndarray, k: np.ndarray) -> np.ndarray:
    H,W = im.shape; pad = k.shape[0]//2
    p = np.pad(im, ((pad,pad),(pad,pad)), mode="reflect")
    out = np.zeros_like(im, np.float32)
    for y in range(H):
        for x in range(W):
            out[y,x] = np.sum(p[y:y+2*pad+1, x:x+2*pad+1]*k)
    return out

# -------- drop edge-touching clouds
def remove_edge_touching_blobs(mask_bin: np.ndarray, connectivity: int = 8, margin: int = 0) -> np.ndarray:
    """
    Returns mask with connected components that touch the image border removed.
    Also optionally strips a 'margin' frame so we won't sample near edges.
    """
    H,W = mask_bin.shape
    mask = mask_bin.astype(bool).copy()
    if margin > 0:
        mask[:margin,:] = False; mask[-margin:,:] = False
        mask[:, :margin] = False; mask[:, -margin:] = False

    visited = np.zeros_like(mask, bool)
    edge_touch = np.zeros_like(mask, bool)
    from collections import deque
    q = deque()
    for x in range(W):
        if mask[0,x]: q.append((0,x))
        if mask[H-1,x]: q.append((H-1,x))
    for y in range(H):
        if mask[y,0]: q.append((y,0))
        if mask[y,W-1]: q.append((y,W-1))
    nbrs4 = [(-1,0),(1,0),(0,-1),(0,1)]
    nbrs8 = nbrs4 + [(-1,-1),(-1,1),(1,-1),(1,1)]
    nbrs = nbrs8 if connectivity==8 else nbrs4
    while q:
        y,x = q.popleft()
        if visited[y,x] or not mask[y,x]: continue
        visited[y,x] = True; edge_touch[y,x] = True
        for dy,dx in nbrs:
            ny,nx = y+dy, x+dx
            if 0<=ny<H and 0<=nx<W and not visited[ny,nx] and mask[ny,nx]:
                q.append((ny,nx))
    return (mask & ~edge_touch).astype(np.float32)

# -------- structure-tensor eigen decomposition
def eig2d(a,b,c):
    tr = a+b; det = a*b - c*c
    s = np.sqrt(np.maximum(tr*tr - 4*det, 0.0))
    l1 = 0.5*(tr + s); l2 = 0.5*(tr - s)
    v1 = np.stack([np.where(np.abs(c)>1e-8, l1-b, 1.0),
                   np.where(np.abs(c)>1e-8, c, 0.0)], -1)
    v1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True)+1e-8)
    v2 = np.stack([-v1[...,1], v1[...,0]], -1)
    return l1,l2,v1,v2

@dataclass
class G2D:
    x: float; y: float
    cov: np.ndarray   # 2x2 (base covariance at z_ref)
    amp: float
    color: np.ndarray # 3
    alpha: float
    z: float          # depth (>0: farther = larger z)

def sample_points(weight: np.ndarray, N: int, seed: int=7) -> Tuple[np.ndarray,np.ndarray]:
    rng = np.random.default_rng(seed)
    w = np.clip(weight,0,None).ravel().astype(np.float64)
    w_sum = w.sum()
    if w_sum <= 0: raise RuntimeError("No sampling mass; loosen thresholds.")
    w /= w_sum
    H,W = weight.shape
    idx = rng.choice(H*W, size=min(N, H*W), replace=False, p=w)
    ys = idx // W; xs = idx % W
    return xs, ys

def render_cloud_only(H:int, W:int, comps: List[G2D], clip_sigma: float, z_ref: float = 2.0, fog: float = 0.0):
    """
    Premultiplied 'over' on black with depth-sorted splats.
    - Sorts by z (far→near).
    - Scales covariance by (z_ref/z)^2 so farther blobs look smaller.
    - Optional depth fog: multiply alpha by exp(-fog * max(0, z - z_ref)).
    Returns (rgb_premul, alpha).
    """
    out_rgb = np.zeros((H,W,3), np.float32)
    out_a   = np.zeros((H,W),    np.float32)

    # far → near: draw larger z first
    comps_sorted = sorted(comps, key=lambda g: g.z, reverse=True)

    for g in comps_sorted:
        z = max(1e-3, float(g.z))
        scale = float(z_ref) / z
        scale = max(scale, 1e-3)
        C = (g.cov * (scale**2)).astype(np.float32)

        # tight bbox from covariance
        vals,_ = np.linalg.eigh(C)
        smax = float(np.sqrt(np.maximum(vals,1e-9)).max())
        ext  = int(clip_sigma * smax) + 2
        x0 = max(0,int(g.x)-ext); x1=min(W-1,int(g.x)+ext)
        y0 = max(0,int(g.y)-ext); y1=min(H-1,int(g.y)+ext)
        if x1<x0 or y1<y0: continue

        xs = np.arange(x0,x1+1, dtype=np.float32) - g.x
        ys = np.arange(y0,y1+1, dtype=np.float32) - g.y
        X,Y = np.meshgrid(xs,ys)
        P = np.stack([X,Y], -1)
        try: invC = np.linalg.inv(C)
        except np.linalg.LinAlgError: invC = np.linalg.pinv(C)
        M = np.einsum("...i,ij,...j->...", P, invC, P)
        G = np.exp(-0.5*M)

        # optional depth fog (constant per gaussian)
        fog_factor = np.exp(-fog * max(0.0, z - z_ref)) if fog > 0.0 else 1.0

        a_src = np.clip(fog_factor * g.alpha * g.amp * G, 0.0, 1.0)   # opacity field
        src_rgb = g.color[None,None,:] * a_src[...,None]              # premultiplied

        # over on black
        a_dst = out_a[y0:y1+1, x0:x1+1]
        rgb_dst = out_rgb[y0:y1+1, x0:x1+1]
        out_rgb[y0:y1+1, x0:x1+1] = src_rgb + rgb_dst * (1 - a_src[...,None])
        out_a  [y0:y1+1, x0:x1+1] = a_src + a_dst * (1 - a_src)

    return np.clip(out_rgb,0,1), np.clip(out_a,0,1)

# --------------------------- pipeline
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Cloud image (RGB).")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--width", type=int, default=1024, help="Resize width (keep aspect).")
    ap.add_argument("--num", type=int, default=12000, help="# Gaussians.")
    ap.add_argument("--val-thresh", type=float, default=0.72, help="HSV V threshold.")
    ap.add_argument("--sat-thresh", type=float, default=0.42, help="HSV S threshold.")
    ap.add_argument("--edge-connectivity", type=int, default=8, choices=[4,8])
    ap.add_argument("--edge-margin", type=int, default=0)
    ap.add_argument("--edge-weight", type=float, default=0.6, help="Sampling weight on edges (0..1).")
    ap.add_argument("--clip-sigma", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=7)

    # ---- depth controls ----
    ap.add_argument("--z-near", type=float, default=1.0, help="Closest depth (>0).")
    ap.add_argument("--z-far",  type=float, default=3.0, help="Farthest depth (> z-near).")
    ap.add_argument(
        "--z-mode",
        choices=["constant", "linear_y", "random", "bright"],  # ← added "bright"
        default="bright",
        help="Depth assignment: constant | linear_y (top=far) | random | bright (brighter=nearer)."
    )
    ap.add_argument("--z-ref",  type=float, default=2.0, help="Reference depth for base cov (scales by z_ref/z).")
    ap.add_argument("--fog",    type=float, default=0.0, help="Depth-fog strength (0 = none).")

    args = ap.parse_args()

    ensure_dir(args.outdir)

    # load & resize
    src = Image.open(args.input).convert("RGB")
    W = args.width; H = int(W * src.height / src.width)
    src = src.resize((W,H))
    rgb = np.asarray(src, np.float32) / 255.0

    # segment clouds (HSV)
    hsv = rgb_to_hsv(rgb)
    SAT, VAL = hsv[...,1], hsv[...,2]
    mask_raw = (VAL > args.val_thresh) & (SAT < args.sat_thresh)
    mask_bin = (box_blur(mask_raw.astype(np.float32), iters=2) > 0.5).astype(np.float32)

    # remove incomplete edge clouds
    mask_interior = remove_edge_touching_blobs(mask_bin, connectivity=args.edge_connectivity,
                                               margin=args.edge_margin)

    # target grayscale for fitting + debug saves
    gray = VAL * (1.0 - 0.6*SAT) * mask_interior
    if gray.max() > 0:
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

    Image.fromarray(to_uint8(mask_interior)).save(os.path.join(args.outdir, "mask.png"))
    Image.fromarray(to_uint8(gray)).save(os.path.join(args.outdir, "gray_for_model.png"))

    # gradients -> structure tensor -> edge-aware sampling weights
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], np.float32)/8.0
    ky = kx.T
    gx, gy = conv2(gray, kx), conv2(gray, ky)
    grad = np.sqrt(gx*gx + gy*gy); grad = grad/(grad.max()+1e-8)
    Jxx, Jyy, Jxy = box_blur(gx*gx,1), box_blur(gy*gy,1), box_blur(gx*gy,1)
    _,_,v1,v2 = eig2d(Jxx,Jyy,Jxy)

    # weights: mix interior mass + edge emphasis
    w = ((1.0 - args.edge_weight) * mask_interior + args.edge_weight * grad) * mask_interior
    w_sum = w.sum()
    if w_sum <= 0: raise RuntimeError("Empty mask; loosen thresholds.")
    w /= w_sum

    # --- AUTO-CAP num if request is too large ---
    support = int((w > 0).sum())
    if support == 0:
        raise RuntimeError("Empty mask; loosen thresholds.")
    if args.num > support:
        print(f"[note] requested --num={args.num}, but only {support} pixels have non-zero mass; capping to {support}.")
        args.num = support

    xs, ys = sample_points(w, args.num, seed=args.seed)

    # depth assignment helper
    # depth assignment helper
    rng_z = np.random.default_rng(args.seed + 12345)
    z_near = max(1e-3, float(args.z_near))
    z_far = max(z_near + 1e-6, float(args.z_far))

    def z_from_mode(xi: int, yi: int) -> float:
        if args.z_mode == "constant":
            return 0.5 * (z_near + z_far)
        elif args.z_mode == "random":
            return float(rng_z.uniform(z_near, z_far))
        elif args.z_mode == "linear_y":
            # top -> far, bottom -> near (invert if you prefer)
            t = 1.0 - (yi / max(1, H - 1))
            return z_near + (z_far - z_near) * t
        else:  # "bright": brighter (denser) -> nearer
            b = float(gray[yi, xi])  # gray in [0,1]
            t = 1.0 - b  # dark -> far, bright -> near
            return z_near + (z_far - z_near) * t

    # build gaussians
    comps: List[G2D] = []
    for x, y in zip(xs, ys):
        e = grad[y, x]
        base = 2.0 + 3.0 * (1.0 - e)  # 2..5 px
        sigma_n = base * 0.9  # across edge
        sigma_t = base * 1.1  # along edge
        ev1, ev2 = v1[y,x], v2[y,x]
        R = np.stack([ev1, ev2], -1)
        S = np.diag([sigma_n**2, sigma_t**2])
        C = (R @ S @ R.T).astype(np.float32)

        amp = float(gray[y,x])           # match local brightness (peak=amp)
        shade = max(0.6, float(gray[y, x]))  # keep some shading
        col = np.array([1.0, 1.0, 1.0], np.float32) * shade
        alpha = float(mask_interior[y,x]) * 0.95

        z = z_from_mode(int(x), int(y))

        comps.append(G2D(float(x), float(y), C, amp, col, alpha, float(z)))

    # render cloud-only (no sky) with depth
    rgb_premul, a = render_cloud_only(H, W, comps, clip_sigma=args.clip_sigma, z_ref=args.z_ref, fog=args.fog)

    # save black background version
    black_rgb = rgb_premul.copy()  # premultiplied over black is already the final RGB
    Image.fromarray(to_uint8(black_rgb)).save(os.path.join(args.outdir, "cloud_only_black.png"))

    # save transparent (un-premultiply for PNG)
    rgba = np.zeros((H,W,4), np.uint8)
    a_clamped = np.clip(a, 0, 1)
    rgb_unpremul = np.zeros_like(rgb_premul, dtype=np.float32)
    np.divide(rgb_premul, a_clamped[..., None],
              out=rgb_unpremul,
              where=(a_clamped[..., None] > 1e-6))
    rgb_unpremul = np.clip(rgb_unpremul, 0, 1)

    rgba[...,0:3] = to_uint8(rgb_unpremul)
    rgba[...,3] = to_uint8(a_clamped)   # alpha channel
    Image.fromarray(rgba, "RGBA").save(os.path.join(args.outdir, "cloud_only_transparent.png"))

    # model (now includes z)
    # model (compact form: arrays, no spaces)
    model = {
        "w": int(W), "h": int(H),

        # Each component is [x, y, z, amp, alpha, [r,g,b], [[c00,c01],[c10,c11]]]
        "c": [
            {"x": g.x, "y": g.y, "z": g.z, "cov": g.cov.tolist(),
             "amplitude": g.amp, "color": g.color.tolist(), "alpha": g.alpha}
            for g in comps
        ],
        "z": float(args.z_ref),
        "fog": float(args.fog),
    }
    with open(os.path.join(args.outdir, "model.json"), "w") as f: json.dump(model, f)

    print("[✓] Saved:", os.path.join(args.outdir, "cloud_only_black.png"))
    print("[✓] Saved:", os.path.join(args.outdir, "cloud_only_transparent.png"))
    print("[✓] Saved:", os.path.join(args.outdir, "mask.png"))
    print("[✓] Saved:", os.path.join(args.outdir, "gray_for_model.png"))
    print("[✓] Saved:", os.path.join(args.outdir, "model.json"))

if __name__ == "__main__":
    main()
