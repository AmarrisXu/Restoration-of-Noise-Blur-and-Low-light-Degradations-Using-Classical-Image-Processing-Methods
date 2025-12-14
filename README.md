
## Noise Removal (`denoise.py`)

Comparison of noise removal methods **NLM** and **BM3D** evaluated on synthetic and real noisy images.

### Methods
- **NLM (Non-Local Means)** — Patch-based denoising using similar patches across the image  
- **BM3D (Block-Matching 3D)** — Collaborative filtering in the 3D transform domain  

### Metrics
- **NIQE** — No-reference naturalness (lower = better)  
- **LPIPS** — Perceptual similarity (lower = better)  
- **Time** — Processing time in ms  
- **Memory** — Memory usage in MB  

### Dataset

#### Synthetic Noise (BSD68 / BSD500)
Download clean datasets; synthetic Gaussian noise is added automatically:
- **BSD68**: https://github.com/smartboy110/denoising-datasets/tree/main  
- **BSD500**: https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500  

Place in:
```

BSD68/
BSD500/

```

#### Real Noise
Place noisy images in:
```

testing_data/noise/

````
Or specify a custom path in `run_evaluation()`.

---

### Usage

#### Run Quantitative Evaluation

---
# Blur Restoration Evaluation

Evaluation of classical blur restoration methods (Wiener Filter and Richardson–Lucy) on synthetic and real motion-blurred images.

This module is part of a classical image restoration project and focuses on recovering images degraded by motion blur using non–learning-based techniques.

## Scripts Overview

The blur restoration pipeline is implemented using the following scripts:

- generate_motion_blur_dataset.py Generates synthetic motion-blurred images from clean images by applying linear motion blur kernels.

- run_deblurring_wiener_rl.py Restores blurred images using Wiener filtering and Richardson–Lucy (RL) deconvolution.

- evaluate_deblurring_metrics.py Computes quantitative evaluation metrics for restored images.

- avg.py Aggregates and averages evaluation results across all test images.

All scripts are executed directly using Python without additional configuration or function-level APIs.

## Methods

- **Wiener Filter**
A classical frequency-domain deconvolution method that estimates the latent sharp image by balancing inverse filtering with noise suppression through a noise-to-signal ratio parameter.

- **Richardson–Lucy (RL) Deconvolution**
An iterative maximum-likelihood deconvolution algorithm based on a Poisson noise assumption. RL progressively refines image estimates and is effective at recovering edges and fine details when the blur kernel is accurate.

## Metrics

- **PSNR** — Pixel-level reconstruction fidelity (higher is better, synthetic blur only)

- **SSIM** — Structural similarity to ground-truth images (higher is better, synthetic blur only)

- **NIQE** — No-reference image naturalness score (lower is better)

- **LPIPS** — Perceptual similarity metric based on deep feature distances (lower is better)

Note: Processing time and memory usage are not evaluated in this blur restoration experiment.

## Dataset
### Synthetic Motion Blur (COCO128)

Clean images are taken from COCO128, a small subset of the COCO dataset containing 128 clean images, and are stored in Google Drive for dataset management.

Before running the scripts, the images are downloaded from Google Drive and placed in a local directory, for example:

```
COCO128/
```


Synthetic motion blur is generated automatically by convolving clean images with linear motion blur kernels of varying lengths and directions.

The COCO128 subset is used to:

- Reduce computational overhead

- Enable fast and reproducible experiments

- Maintain sufficient scene diversity for blur restoration evaluation



Notes on Data Access

Google Drive is used only as a dataset storage and sharing platform

All scripts operate on local file paths after download

No online access or Drive API is required during execution

How to Run

All scripts are executed directly using Python:

python generate_motion_blur_dataset.py
python run_deblurring_wiener_rl.py
python evaluate_deblurring_metrics.py
python avg.py

## Output

- Synthetic blurred images and corresponding restored results

- Quantitative evaluation results for each image

- Averaged metrics summarizing overall restoration performance

## Notes

### Wiener Filter

- Simple and computationally efficient

- Suitable for mild motion blur

- Tends to oversmooth edges

### Richardson–Lucy

- Strong edge and detail recovery

- Sensitive to blur kernel accuracy

- Excessive iterations may introduce ringing artifacts

---

# Low-Light Image Enhancement Evaluation

Comparison of low-light enhancement methods (MSR, MSRCR, CLAHE) evaluated on the ExDark dataset.

## Methods
- **MSR** (Multi-Scale Retinex) - Estimates illumination using multi-scale Gaussian filtering
- **MSRCR** (MSR with Color Restoration) - Adds color correction to MSR
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) - Local histogram equalization with contrast limiting

## Metrics
- **NIQE** - No-reference image quality (lower = more natural)
- **LPIPS** - Perceptual similarity (lower = better preservation)
- **Time** - Processing time in milliseconds
- **Memory** - Memory usage in MB

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Download the ExDark dataset from [GitHub](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) and extract to `ExDark/` folder.

## Usage

### Run Quantitative Evaluation
```python
from lowlight import run_evaluation

# Evaluate on 50 samples
summary = run_evaluation(num_samples=50)

# Evaluate with custom dataset path
summary = run_evaluation(num_samples=100, dataset_path="path/to/ExDark")
```

### Run Visual Comparison
```python
from lowlight import run_visual_comparison

# Generate comparison figure for a specific image
fig = run_visual_comparison("ExDark/Cat/2015_03068.jpg")

# Save to custom path without displaying
fig = run_visual_comparison("my_image.jpg", save_path="result.png", show=False)
```

### Run from Command Line
```bash
python lowlight.py
```

## Output
- Prints metrics table to console
- Saves `lowlight_comparison.png` with side-by-side visual comparison

---

