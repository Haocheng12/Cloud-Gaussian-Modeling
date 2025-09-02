# Cloud-only 2D Gaussian Splatting

This project provides a Python script (`to2dgs.py`) for extracting and rendering clouds from an image using **2D Gaussian splatting with depth**.

The script outputs cloud-only renderings (black background and transparent) as well as a Gaussian model in JSON format for further visualization or processing.

---

## ✨ Features

- **Cloud segmentation (HSV space)**  
  - Bright + low-saturation pixels are extracted as clouds.

- **Edge-aware filtering**  
  - Removes incomplete edge-touching blobs.  
  - Structure tensor produces anisotropic covariances aligned with edges.

- **Depth assignment (z-axis)**  
  - Modes:  
    - `constant` – same depth everywhere  
    - `linear_y` – top = far, bottom = near  
    - `random` – random depths  
    - `bright` – brighter = nearer (default)  
  - Depth controls occlusion, scale, and optional fog fading.

- **Rendering outputs**  
  - `cloud_only_black.png` – clouds on black  
  - `cloud_only_transparent.png` – RGBA with transparency  
  - `mask.png` – binary segmentation mask  
  - `gray_for_model.png` – grayscale fitting target  
  - `model.json` – Gaussian parameters `{x, y, z, cov, amplitude, color, alpha}`

---

## 🚀 Installation

Requires **Python 3.8+**.

```bash
pip install numpy pillow


⚙️ Arguments

--input (required): Input RGB image (cloud photo)

--outdir (default: out): Output directory

--width (default: 1024): Resize width (aspect preserved)

--num (default: 12000): Number of Gaussians to sample

--val-thresh (default: 0.72): HSV brightness threshold

--sat-thresh (default: 0.42): HSV saturation threshold

--edge-connectivity (default: 8): Connectivity for edge removal (4 or 8)

--edge-margin (default: 0): Margin to drop near-border pixels

--edge-weight (default: 0.6): Balance between interior and edge emphasis (0–1)

--clip-sigma (default: 4.0): Gaussian cutoff in sigmas

--seed (default: 7): Random seed for reproducibility

--z-near (default: 1.0): Closest depth (>0)

--z-far (default: 3.0): Farthest depth (> z-near)

--z-mode (default: bright): Depth mode (constant, linear_y, random, bright)

--z-ref (default: 2.0): Reference depth for scaling covariance

--fog (default: 0.0): Depth fog strength (0 = none)
