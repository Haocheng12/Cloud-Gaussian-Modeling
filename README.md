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
python to2dgs.py --input <image.png> [options]


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

📂 Output Files

cloud_only_black.png → Rendered clouds on black background

cloud_only_transparent.png → Transparent PNG (RGBA) for compositing

mask.png → Binary segmentation mask of clouds

gray_for_model.png → Grayscale fitting guide for model construction

model.json → Gaussian splatting model with depth info









# Blender Import Script for 2D Gaussian Splatting Models

This script (`import.py`) is designed to **import Gaussian splatting cloud models** (generated from `to2dgs.py`) into **Blender**.  
It reconstructs the splats as instanced quads with correct **position, orientation, scale, color, and transparency** using **Geometry Nodes** and a custom shader.

---

## ✨ Features

- **Load Gaussian models from JSON**
  - Reads the `model.json` produced by the cloud splatting script.
  - Handles width, height, depth, fog, and per-Gaussian attributes.

- **Automatic quad + UV creation**
  - Builds a helper quad mesh with UV coordinates for instancing.

- **Custom material**
  - Shader uses emission + transparency.
  - Reads per-point attributes (`amp`, `alpha_final`, `col`) for rendering.

- **Geometry Nodes instancing**
  - Creates a **Geometry Nodes network** (`GN_GaussianSprites`) to:
    - Instance quads at Gaussian positions.
    - Rotate & scale quads based on covariance.
    - Apply per-Gaussian attributes via vertex groups & color layers.
    - Realize instances for material access.

- **Attribute encoding**
  - Gaussian parameters stored as vertex groups and mesh color attributes:
    - `rad_a` – ellipse radius (major axis)  
    - `rad_b` – ellipse radius (minor axis)  
    - `theta` – rotation angle  
    - `amp` – amplitude  
    - `alpha_final` – alpha (with fog applied)  
    - `col` – RGB color  

- **Blender 4.5 compatible**
  - Uses updated Geometry Nodes and attribute API.

---

## 📂 File Inputs

- **`model.json`** – Produced by [`to2dgs.py`](../to2dgs.py), containing:
  - Canvas dimensions (`w`, `h`)  
  - Reference depth (`z`) and fog factor (`fog`)  
  - List of Gaussian components (`c`), each with:
    - `x`, `y`, `z` – position  
    - `cov` – 2×2 covariance  
    - `amplitude` – brightness weight  
    - `alpha` – transparency  
    - `color` – RGB  

---

## ⚙️ Configuration

At the top of `import.py`, adjust paths and settings:

```python
# ------------------ CONFIG ------------------
model_path  = "C:/Users/OneDrive/Desktop/2dgs/out5/model.json"
units_width = 2.0   # world-space width for model
sigma_clip  = 3.0   # ellipse radius multiplier
use_cycles  = False # use Cycles or Eevee
# --------------------------------------------

