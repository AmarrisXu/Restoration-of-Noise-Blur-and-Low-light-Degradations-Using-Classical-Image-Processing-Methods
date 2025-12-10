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

