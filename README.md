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
```python
from denoise import run_evaluation

# Real noisy images
summary = run_evaluation(num_samples=50)

# Custom real-noise dataset
summary = run_evaluation(num_samples=100, dataset_path="path/to/noisy/images")

# Synthetic BSD noise (Gaussian σ = 10, 25, 50)
summary = run_evaluation(
    num_samples=50,
    use_synthetic=True,
    noise_levels=[10, 25, 50]
)
````

#### Run Visual Comparison

```python
from denoise import run_visual_comparison

# Generate a comparison figure
fig = run_visual_comparison("testing_data/noise/1.jpg")

# Save without displaying
fig = run_visual_comparison("noisy.jpg", save_path="result.png", show=False)
```

#### Generate Output Samples

```python
from denoise import generate_output_samples

generate_output_samples(
    ["img1.jpg", "img2.jpg"],
    output_dir="output_sample/denoise"
)
```

#### Command Line

```bash
python denoise.py
```

---

### Output

* Prints metrics table to console
* Saves `denoise_comparison.png`
* Generates comparison figures in `output_sample/denoise/`

