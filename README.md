# 🌥 Cloud-only 2D Gaussian Splatting & Blender Import

This project provides two Python scripts:

- **`to2dgs.py`** → Extracts and renders clouds from an image using **2D Gaussian splatting with depth**.  
- **`import.py`** → Imports the generated Gaussian model into **Blender** as instanced quads using **Geometry Nodes** and a custom shader.

---

## ✨ Features

### `to2dgs.py`
- **Cloud segmentation (HSV space)** → extracts bright + low-saturation regions as clouds.  
- **Edge-aware filtering** → removes incomplete edge-touching blobs.  
- **Depth assignment (z-axis)** → supports `constant`, `linear_y`, `random`, and `bright` (default).  
- **Rendering outputs** → black background, transparent PNG, mask, grayscale fitting image, and a JSON model.  

### `import.py`
- **Loads Gaussian models from JSON** → reads the `model.json` produced by `to2dgs.py`.  
- **Automatic quad + UV creation** → helper mesh for instancing.  
- **Custom material** → emission + transparency, driven by per-point attributes.  
- **Geometry Nodes instancing** → builds a `GN_GaussianSprites` node tree for instancing quads with correct scale/rotation.  
- **Attribute encoding** → Gaussian data stored as vertex groups & color attributes (`rad_a`, `rad_b`, `theta`, `amp`, `alpha_final`, `col`).  
- **Blender 4.5 compatible**.  

---

## 🚀 Installation

### Python requirements (for `to2dgs.py`)
```bash
pip install numpy pillow
```

### Blender requirements (for `import.py`)
- Blender **4.5+** (Geometry Nodes & attribute API compatibility).  

---

## 🖥 Usage

### 1. Generate Gaussian model
```bash
python to2dgs.py --input <image.png> [options]
```

#### Arguments
- `--input` (required): Input RGB image (cloud photo)  
- `--outdir` (default: `out`): Output directory  
- `--width` (default: `1024`): Resize width (aspect preserved)  
- `--num` (default: `12000`): Number of Gaussians to sample  
- `--val-thresh` (default: `0.72`): HSV brightness threshold  
- `--sat-thresh` (default: `0.42`): HSV saturation threshold  
- `--edge-connectivity` (default: `8`): Connectivity for edge removal (`4` or `8`)  
- `--edge-margin` (default: `0`): Margin to drop near-border pixels  
- `--edge-weight` (default: `0.6`): Balance between interior and edge emphasis (0–1)  
- `--clip-sigma` (default: `4.0`): Gaussian cutoff in sigmas  
- `--seed` (default: `7`): Random seed for reproducibility  
- `--z-near` (default: `1.0`): Closest depth (>0)  
- `--z-far` (default: `3.0`): Farthest depth (> z-near)  
- `--z-mode` (default: `bright`): Depth mode (`constant`, `linear_y`, `random`, `bright`)  
- `--z-ref` (default: `2.0`): Reference depth for scaling covariance  
- `--fog` (default: `0.0`): Depth fog strength (0 = none)  

#### Output files
- **`cloud_only_black.png`** → clouds on black background  
- **`cloud_only_transparent.png`** → transparent PNG (RGBA)  
- **`mask.png`** → binary cloud mask  
- **`gray_for_model.png`** → grayscale fitting guide  
- **`model.json`** → Gaussian splatting model with depth  

---

### 2. Import into Blender
1. Open Blender **4.5+**.  
2. Go to **Scripting → New Script**, load `import.py`.  
3. Edit the config at the top of the file:  
   ```python
   # ------------------ CONFIG ------------------
   model_path  = "C:/path/to/out/model.json"
   units_width = 2.0   # world-space width
   sigma_clip  = 3.0   # ellipse size multiplier
   use_cycles  = False # render engine
   # --------------------------------------------
   ```
4. Run the script.  
5. A new object `CloudGaussians` will appear with instanced quads.  

---

## 📌 Workflow Example

1. Generate model JSON from an image:
   ```bash
   python to2dgs.py --input clouds.jpg --outdir out5
   ```
2. Import into Blender:
   - Update `model_path` in `import.py` → `out5/model.json`  
   - Run script inside Blender  
   - Switch to **Rendered View** to preview the emission shader  

---

## 📂 Example `model.json`

Here’s a simplified example of the JSON format produced by `to2dgs.py`:

```json
{
  "w": 800,
  "h": 600,
  "z": 2.0,
  "fog": 0.0,
  "c": [
    {
      "x": 120.5,
      "y": 240.3,
      "z": 1.8,
      "cov": [[4.2, 0.1], [0.1, 3.7]],
      "amplitude": 0.9,
      "color": [1.0, 1.0, 1.0],
      "alpha": 0.85
    }
  ]
}
```

- `w`, `h` → canvas size  
- `z` → reference depth  
- `fog` → fog factor  
- `c` → list of Gaussian components with position, covariance, amplitude, color, and alpha  

---

## 📜 License

MIT License (modify as needed).
