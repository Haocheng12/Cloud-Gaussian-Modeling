# Cloud-Gaussian-Modeling

Cloud-only 2D Gaussian Splatting

This repository contains a Python script (to2dgs.py) that extracts and renders clouds from an image using 2D Gaussian splatting with depth information.

The output includes cloud-only renderings with alpha transparency and a compact Gaussian model (model.json) suitable for further visualization or processing.

Features

Cloud segmentation in HSV space

Bright, low-saturation regions are extracted as clouds.

Edge-aware processing

Drops incomplete, edge-touching blobs to avoid partial artifacts.

Structure tensor analysis yields anisotropic Gaussian covariances aligned with edges.

Depth assignment (z-axis)

Supports multiple modes:

constant – uniform depth

linear_y – top = far, bottom = near

random – random depth per component

bright – brighter = nearer (default)

Depth influences occlusion order, Gaussian scale, and optional fog fading.

Rendering

Outputs:

cloud_only_black.png – black background

cloud_only_transparent.png – RGBA with transparency

mask.png – binary cloud mask

gray_for_model.png – grayscale target for fitting

model.json – Gaussian parameters {x, y, z, cov, amplitude, color, alpha}

Installation

Requires Python 3.8+.

pip install numpy pillow

Usage
python to2dgs.py --input <image.png> [options]

Arguments
Argument	Default	Description
--input	(req)	Input RGB image (cloud photo).
--outdir	out	Output directory.
--width	1024	Resize width (aspect preserved).
--num	12000	Number of Gaussians to sample.
--val-thresh	0.72	HSV V (brightness) threshold.
--sat-thresh	0.42	HSV S (saturation) threshold.
--edge-connectivity	8	Connectivity for edge removal (4 or 8).
--edge-margin	0	Margin to drop near-border pixels.
--edge-weight	0.6	Balance between interior and edge emphasis (0–1).
--clip-sigma	4.0	Gaussian cutoff in sigmas.
--seed	7	Random seed for reproducibility.
--z-near	1.0	Closest depth (>0).
--z-far	3.0	Farthest depth (> z-near).
--z-mode	bright	Depth mode: constant, linear_y, random, bright.
--z-ref	2.0	Reference depth for scaling covariance.
--fog	0.0	Depth fog strength (0 = none).
Example
python to2dgs.py --input clouds.jpg --outdir results --width 800 --num 8000 --z-mode linear_y --fog 0.3


This will:

Resize the image to width 800px.

Sample ~8000 Gaussians.

Assign depth linearly (top = far, bottom = near).

Apply light fog fading.

Save results in results/.

Output Files

cloud_only_black.png – Render on black background.

cloud_only_transparent.png – Transparent PNG for compositing.

mask.png – Binary segmentation mask.

gray_for_model.png – Grayscale fitting guide.

model.json – Gaussian splatting model (with depth).
