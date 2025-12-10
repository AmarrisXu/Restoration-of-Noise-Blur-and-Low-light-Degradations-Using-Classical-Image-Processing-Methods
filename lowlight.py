"""
Low-Light Image Enhancement: MSR, MSRCR, and CLAHE
Evaluation on ExDark Dataset with NIQE, LPIPS metrics
"""

import os
import cv2
import numpy as np
import time
import tracemalloc
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# For NIQE gamma function
from scipy.special import gamma as scipy_gamma
from scipy.ndimage import convolve
from scipy.stats import genpareto

# For LPIPS and NIQE
try:
    import torch
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: torch/lpips not installed. LPIPS metric will be unavailable.")
    print("Install with: pip install torch lpips")

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage import img_as_float
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# ============================================================================
# METHOD A: RETINEX (MSR / MSRCR)
# ============================================================================

class RetinexEnhancer:
    """
    Multi-Scale Retinex (MSR) and Multi-Scale Retinex with Color Restoration (MSRCR)
    Based on Retinex theory - estimates and manipulates illumination component.
    Optimized for better performance using vectorized operations.
    """
    
    def __init__(self, sigma_list: List[int] = [15, 80, 250]):
        self.sigma_list = sigma_list
    
    def _single_scale_retinex(self, img: np.ndarray, sigma: int) -> np.ndarray:
        """
        Single Scale Retinex (SSR) - Optimized version.
        R(x,y) = log(I(x,y)) - log(I(x,y) * G(x,y))
        """
        # Avoid log(0) - use maximum in-place for efficiency
        img_safe = np.maximum(img, 1.0)
        
        # Compute kernel size (must be odd, at least 6*sigma)
        ksize = int(sigma * 6) | 1  # Make odd
        ksize = max(ksize, 3)
        
        # Gaussian blur to estimate illumination
        blur = cv2.GaussianBlur(img_safe, (ksize, ksize), sigma)
        blur = np.maximum(blur, 1.0)
        
        # Retinex: log(image) - log(illumination)
        # Using natural log for speed, then scaling
        retinex = np.log(img_safe) - np.log(blur)
        
        return retinex
    
    def msr(self, img: np.ndarray) -> np.ndarray:
        """
        Multi-Scale Retinex (MSR) - Optimized version.
        Combines multiple SSR results at different scales.
        """
        img_float = img.astype(np.float32) + 1.0  # float32 is faster
        
        # Apply MSR to all channels at once
        retinex = np.zeros_like(img_float)
        for sigma in self.sigma_list:
            retinex += self._single_scale_retinex(img_float, sigma)
        
        retinex /= len(self.sigma_list)
        
        # Vectorized normalization for all channels
        retinex = self._normalize_all_channels(retinex)
        
        return retinex.astype(np.uint8)
    
    def msrcr(self, img: np.ndarray, alpha: float = 125.0, beta: float = 46.0,
              gain: float = 192.0, offset: float = -30.0) -> np.ndarray:
        """
        Multi-Scale Retinex with Color Restoration (MSRCR) - Optimized.
        Adds color restoration term to correct color distortions.
        """
        img_float = img.astype(np.float32) + 1.0
        
        # Multi-scale retinex
        retinex = np.zeros_like(img_float)
        for sigma in self.sigma_list:
            retinex += self._single_scale_retinex(img_float, sigma)
        retinex /= len(self.sigma_list)
        
        # Color restoration function: C(x,y) = beta * (log(alpha * I_i) - log(sum(I)))
        img_sum = np.sum(img_float, axis=2, keepdims=True)
        img_sum = np.maximum(img_sum, 1.0)
        
        color_restoration = beta * (np.log(alpha * img_float) - np.log(img_sum))
        
        # Apply color restoration
        msrcr = gain * (color_restoration * retinex + offset)
        
        # Vectorized normalization
        msrcr = self._normalize_all_channels(msrcr)
        
        return np.clip(msrcr, 0, 255).astype(np.uint8)
    
    def _normalize_all_channels(self, img: np.ndarray) -> np.ndarray:
        """Normalize all channels to [0, 255] efficiently."""
        result = np.zeros_like(img)
        for i in range(3):
            channel = img[:, :, i]
            c_min, c_max = channel.min(), channel.max()
            if c_max - c_min > 0:
                result[:, :, i] = 255.0 * (channel - c_min) / (c_max - c_min)
        return result
    
    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 255] range."""
        img_min = img.min()
        img_max = img.max()
        
        if img_max - img_min == 0:
            return np.zeros_like(img)
        
        return 255.0 * (img - img_min) / (img_max - img_min)


# ============================================================================
# METHOD B: CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ============================================================================

class CLAHEEnhancer:
    """
    CLAHE - Partitions image into tiles and performs histogram equalization
    with contrast limiting to avoid noise amplification.
    """
    
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def enhance(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE enhancement.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def enhance_rgb(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to each RGB channel separately.
        """
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        
        channels = cv2.split(img)
        enhanced_channels = [clahe.apply(ch) for ch in channels]
        
        return cv2.merge(enhanced_channels)


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class NIQE:
    """
    Natural Image Quality Evaluator (NIQE) - No-reference quality metric.
    Lower scores indicate better quality (more natural appearance).
    """
    
    def __init__(self):
        self.patch_size = 96
        self.stride = 32
        # Pre-compute gamma lookup table for efficiency
        self.gam = np.arange(0.2, 10.001, 0.001)
        self.r_gam = (scipy_gamma(2.0/self.gam)**2) / (scipy_gamma(1.0/self.gam) * scipy_gamma(3.0/self.gam))
    
    def _estimate_ggd_params(self, vec: np.ndarray) -> Tuple[float, float]:
        """Estimate Generalized Gaussian Distribution parameters (shape, variance)."""
        vec = vec.flatten()
        vec = vec[~np.isnan(vec)]
        
        if len(vec) == 0:
            return 1.0, 1.0
        
        sigma_sq = np.mean(vec ** 2)
        if sigma_sq < 1e-10:
            return 1.0, 1e-10
            
        E_abs = np.mean(np.abs(vec))
        if E_abs < 1e-10:
            return 1.0, sigma_sq
            
        rho = sigma_sq / (E_abs ** 2 + 1e-10)
        
        # Find closest gamma value
        diff = np.abs(self.r_gam - rho)
        idx = np.argmin(diff)
        shape = self.gam[idx]
        
        return shape, sigma_sq
    
    def _estimate_aggd_params(self, vec: np.ndarray) -> Tuple[float, float, float, float]:
        """Estimate Asymmetric Generalized Gaussian Distribution parameters."""
        vec = vec.flatten()
        vec = vec[~np.isnan(vec)]
        
        if len(vec) == 0:
            return 1.0, 0.1, 0.1, 0.0
        
        # Separate left and right
        left = vec[vec < 0]
        right = vec[vec >= 0]
        
        left_std = np.sqrt(np.mean(left ** 2)) if len(left) > 0 else 1e-10
        right_std = np.sqrt(np.mean(right ** 2)) if len(right) > 0 else 1e-10
        
        # Avoid division by zero
        gamma_hat = left_std / (right_std + 1e-10)
        
        # Compute r-hat
        mean_abs = np.mean(np.abs(vec))
        mean_sq = np.mean(vec ** 2)
        
        if mean_sq < 1e-10:
            return 1.0, left_std, right_std, 0.0
            
        rhat = (mean_abs ** 2) / (mean_sq + 1e-10)
        
        # Normalize r-hat
        rhat_norm = rhat * (gamma_hat ** 3 + 1) * (gamma_hat + 1) / ((gamma_hat ** 2 + 1) ** 2 + 1e-10)
        
        # Find alpha
        diff = np.abs(self.r_gam - rhat_norm)
        idx = np.argmin(diff)
        alpha = self.gam[idx]
        
        # Mean
        mean_val = np.mean(vec)
        
        return alpha, left_std, right_std, mean_val
    
    def _compute_mscn(self, img: np.ndarray) -> np.ndarray:
        """Compute Mean Subtracted Contrast Normalized (MSCN) coefficients."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = img.astype(np.float64)
        
        # Local mean using Gaussian filter
        kernel_size = 7
        mu = cv2.GaussianBlur(img, (kernel_size, kernel_size), 7/6)
        
        # Local variance
        mu_sq = mu ** 2
        sigma = np.sqrt(np.maximum(cv2.GaussianBlur(img ** 2, (kernel_size, kernel_size), 7/6) - mu_sq, 0))
        
        # MSCN coefficients
        mscn = (img - mu) / (sigma + 1.0)
        
        return mscn
    
    def _compute_features(self, img: np.ndarray) -> np.ndarray:
        """Compute NIQE features from image."""
        mscn = self._compute_mscn(img)
        
        h, w = mscn.shape
        features_list = []
        
        # Extract patches
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                patch = mscn[i:i+self.patch_size, j:j+self.patch_size]
                
                # Check if patch has enough variance (skip flat regions)
                if np.std(patch) < 0.01:
                    continue
                
                feat = self._patch_features(patch)
                if feat is not None and not np.any(np.isnan(feat)):
                    features_list.append(feat)
        
        if len(features_list) == 0:
            # Fallback: compute features on whole image
            feat = self._patch_features(mscn)
            if feat is not None:
                return feat
            return np.ones(18) * 5.0  # Default score
        
        return np.mean(features_list, axis=0)
    
    def _patch_features(self, patch: np.ndarray) -> np.ndarray:
        """Compute 18-dim feature vector for a patch."""
        try:
            features = []
            
            # 1. GGD shape and variance (2 features)
            shape, var = self._estimate_ggd_params(patch)
            features.extend([shape, var])
            
            # 2. Paired products in 4 directions (4 x 4 = 16 features)
            shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
            
            for dy, dx in shifts:
                shifted = np.roll(np.roll(patch, dy, axis=0), dx, axis=1)
                paired = patch * shifted
                
                alpha, left_std, right_std, mean_val = self._estimate_aggd_params(paired)
                features.extend([alpha, mean_val, left_std ** 2, right_std ** 2])
            
            return np.array(features)
        except Exception:
            return None
    
    def calculate(self, img: np.ndarray) -> float:
        """
        Calculate NIQE score. Lower is better.
        """
        features = self._compute_features(img)
        
        # Reference statistics from natural images (simplified model)
        # These are approximate values based on natural image statistics
        ref_mean = np.array([2.5, 0.5, 2.0, 0.0, 0.3, 0.3,
                           2.0, 0.0, 0.3, 0.3, 2.0, 0.0, 0.3, 0.3,
                           2.0, 0.0, 0.3, 0.3])
        ref_std = np.array([0.5, 0.3, 0.5, 0.2, 0.2, 0.2,
                          0.5, 0.2, 0.2, 0.2, 0.5, 0.2, 0.2, 0.2,
                          0.5, 0.2, 0.2, 0.2])
        
        # Ensure feature vector matches expected size
        if len(features) != len(ref_mean):
            features = np.resize(features, len(ref_mean))
        
        # Compute distance from natural image model
        # Using Mahalanobis-like distance (simplified)
        diff = (features - ref_mean) / (ref_std + 1e-10)
        score = np.sqrt(np.mean(diff ** 2))
        
        # Scale to typical NIQE range (roughly 2-8 for most images)
        score = max(1.0, min(score, 15.0))
        
        return float(score)


class LPIPSCalculator:
    """LPIPS - Learned Perceptual Image Patch Similarity."""
    
    def __init__(self):
        if LPIPS_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        else:
            self.loss_fn = None
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate LPIPS between two images. Lower is better (more similar).
        """
        if not LPIPS_AVAILABLE:
            return -1.0
        
        # Convert BGR to RGB and normalize to [-1, 1]
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        img1_tensor = torch.from_numpy(img1_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        img2_tensor = torch.from_numpy(img2_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        
        # Ensure same size
        if img1_tensor.shape != img2_tensor.shape:
            h = min(img1_tensor.shape[2], img2_tensor.shape[2])
            w = min(img1_tensor.shape[3], img2_tensor.shape[3])
            img1_tensor = img1_tensor[:, :, :h, :w]
            img2_tensor = img2_tensor[:, :, :h, :w]
        
        img1_tensor = img1_tensor.to(self.device)
        img2_tensor = img2_tensor.to(self.device)
        
        with torch.no_grad():
            distance = self.loss_fn(img1_tensor, img2_tensor)
        
        return float(distance.cpu().numpy())


# ============================================================================
# EXDARK DATASET HANDLER
# ============================================================================

class ExDarkDataset:
    """Handler for ExDark (Exclusively Dark) dataset."""
    
    CATEGORIES = [
        'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair',
        'Cup', 'Dog', 'Motorbike', 'People', 'Table'
    ]
    
    def __init__(self, root_path: str):
        """
        Initialize ExDark dataset.
        """
        self.root_path = Path(root_path)
        self.images = []
        self._scan_dataset()
    
    def _scan_dataset(self):
        """Scan dataset directory for images."""
        # ExDark structure: ExDark/images/Category/image.jpg
        # or ExDark/Category/image.jpg
        
        possible_paths = [
            self.root_path / 'ExDark',
            self.root_path / 'images',
            self.root_path,
        ]
        
        for base_path in possible_paths:
            if not base_path.exists():
                continue
                
            for category in self.CATEGORIES:
                cat_path = base_path / category
                if cat_path.exists():
                    for img_path in cat_path.glob('*'):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            self.images.append({
                                'path': str(img_path),
                                'category': category,
                                'name': img_path.name
                            })
        
        # If structured search fails, try flat directory
        if not self.images:
            for img_path in self.root_path.rglob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.append({
                        'path': str(img_path),
                        'category': 'Unknown',
                        'name': img_path.name
                    })
        
        print(f"Found {len(self.images)} images in dataset")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get image by index."""
        info = self.images[idx]
        img = cv2.imread(info['path'])
        return {
            'image': img,
            'path': info['path'],
            'category': info['category'],
            'name': info['name']
        }
    
    def get_sample(self, n: int = 100) -> List[Dict]:
        """Get a random sample of images."""
        indices = np.random.choice(len(self.images), min(n, len(self.images)), replace=False)
        return [self[i] for i in indices]


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

class LowLightEvaluator:
    """Comprehensive evaluator for low-light enhancement methods."""
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize evaluator.
        """
        # Initialize enhancers
        self.retinex = RetinexEnhancer()
        self.clahe = CLAHEEnhancer()
        
        # Initialize metrics
        self.niqe = NIQE()
        self.lpips = LPIPSCalculator()
        
        # Dataset
        if dataset_path and os.path.exists(dataset_path):
            self.dataset = ExDarkDataset(dataset_path)
        else:
            self.dataset = None
            print("Dataset not found. Will use sample images or create synthetic ones.")
        
        # Results storage
        self.results = {
            'msr': {'niqe': [], 'lpips': [], 'time': [], 'memory': []},
            'msrcr': {'niqe': [], 'lpips': [], 'time': [], 'memory': []},
            'clahe': {'niqe': [], 'lpips': [], 'time': [], 'memory': []},
            'original': {'niqe': [], 'lpips': [], 'time': [], 'memory': []}
        }
    
    def create_synthetic_lowlight(self, img: np.ndarray, gamma: float = 2.5) -> np.ndarray:
        """Create synthetic low-light image from normal image."""
        # Gamma correction to darken
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(img, table)
    
    def measure_performance(self, func, img: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Measure time and memory usage of an enhancement function.
        """
        tracemalloc.start()
        start_time = time.perf_counter()
        
        result = func(img)
        
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        time_ms = (end_time - start_time) * 1000
        memory_mb = peak / (1024 * 1024)
        
        return result, time_ms, memory_mb
    
    def enhance_image(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply all enhancement methods to an image."""
        return {
            'original': img.copy(),
            'msr': self.retinex.msr(img),
            'msrcr': self.retinex.msrcr(img),
            'clahe': self.clahe.enhance(img)
        }
    
    def evaluate_single(self, img: np.ndarray, reference: np.ndarray = None) -> Dict:
        """
        Evaluate all methods on a single image.
        """
        results = {}
        
        # Original
        niqe_orig = self.niqe.calculate(img)
        results['original'] = {
            'niqe': niqe_orig,
            'lpips': 0.0,
            'time': 0.0,
            'memory': 0.0
        }
        
        # MSR
        msr_result, msr_time, msr_mem = self.measure_performance(self.retinex.msr, img)
        niqe_msr = self.niqe.calculate(msr_result)
        lpips_msr = self.lpips.calculate(img, msr_result) if reference is None else self.lpips.calculate(reference, msr_result)
        results['msr'] = {
            'niqe': niqe_msr,
            'lpips': lpips_msr,
            'time': msr_time,
            'memory': msr_mem,
            'image': msr_result
        }
        
        # MSRCR
        msrcr_result, msrcr_time, msrcr_mem = self.measure_performance(self.retinex.msrcr, img)
        niqe_msrcr = self.niqe.calculate(msrcr_result)
        lpips_msrcr = self.lpips.calculate(img, msrcr_result) if reference is None else self.lpips.calculate(reference, msrcr_result)
        results['msrcr'] = {
            'niqe': niqe_msrcr,
            'lpips': lpips_msrcr,
            'time': msrcr_time,
            'memory': msrcr_mem,
            'image': msrcr_result
        }
        
        # CLAHE
        clahe_result, clahe_time, clahe_mem = self.measure_performance(self.clahe.enhance, img)
        niqe_clahe = self.niqe.calculate(clahe_result)
        lpips_clahe = self.lpips.calculate(img, clahe_result) if reference is None else self.lpips.calculate(reference, clahe_result)
        results['clahe'] = {
            'niqe': niqe_clahe,
            'lpips': lpips_clahe,
            'time': clahe_time,
            'memory': clahe_mem,
            'image': clahe_result
        }
        
        return results
    
    def evaluate_dataset(self, num_samples: int = 50) -> Dict:
        """
        Evaluate methods on dataset samples.
        """
        if self.dataset is None or len(self.dataset) == 0:
            print("No dataset available. Creating synthetic test images...")
            return self._evaluate_synthetic(num_samples)
        
        samples = self.dataset.get_sample(num_samples)
        
        all_results = []
        for i, sample in enumerate(samples):
            if sample['image'] is None:
                continue
            
            print(f"Processing image {i+1}/{len(samples)}: {sample['name']}")
            
            result = self.evaluate_single(sample['image'])
            all_results.append(result)
            
            # Store in results
            for method in ['msr', 'msrcr', 'clahe', 'original']:
                self.results[method]['niqe'].append(result[method]['niqe'])
                self.results[method]['time'].append(result[method]['time'])
                self.results[method]['memory'].append(result[method]['memory'])
                if 'lpips' in result[method]:
                    self.results[method]['lpips'].append(result[method]['lpips'])
        
        return self._aggregate_results()
    
    def _evaluate_synthetic(self, num_samples: int) -> Dict:
        """Evaluate on synthetic low-light images."""
        # Create a sample image
        sample_img = np.random.randint(100, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some structure
        cv2.rectangle(sample_img, (100, 100), (300, 300), (200, 150, 100), -1)
        cv2.circle(sample_img, (400, 250), 80, (100, 200, 150), -1)
        
        for i in range(num_samples):
            # Create variation
            noise = np.random.randint(-20, 20, sample_img.shape, dtype=np.int16)
            varied = np.clip(sample_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Create low-light version
            low_light = self.create_synthetic_lowlight(varied)
            
            print(f"Processing synthetic image {i+1}/{num_samples}")
            
            result = self.evaluate_single(low_light, reference=varied)
            
            for method in ['msr', 'msrcr', 'clahe', 'original']:
                self.results[method]['niqe'].append(result[method]['niqe'])
                self.results[method]['time'].append(result[method]['time'])
                self.results[method]['memory'].append(result[method]['memory'])
                if 'lpips' in result[method]:
                    self.results[method]['lpips'].append(result[method]['lpips'])
        
        return self._aggregate_results()
    
    def _aggregate_results(self) -> Dict:
        """Aggregate results into summary statistics."""
        summary = {}
        
        for method in ['msr', 'msrcr', 'clahe', 'original']:
            summary[method] = {
                'niqe_mean': np.mean(self.results[method]['niqe']) if self.results[method]['niqe'] else 0,
                'niqe_std': np.std(self.results[method]['niqe']) if self.results[method]['niqe'] else 0,
                'lpips_mean': np.mean([x for x in self.results[method]['lpips'] if x >= 0]) if self.results[method]['lpips'] else 0,
                'lpips_std': np.std([x for x in self.results[method]['lpips'] if x >= 0]) if self.results[method]['lpips'] else 0,
                'time_mean': np.mean(self.results[method]['time']) if self.results[method]['time'] else 0,
                'time_std': np.std(self.results[method]['time']) if self.results[method]['time'] else 0,
                'memory_mean': np.mean(self.results[method]['memory']) if self.results[method]['memory'] else 0,
                'memory_std': np.std(self.results[method]['memory']) if self.results[method]['memory'] else 0,
            }
        
        return summary
    
    def create_comparison_figure(self, img: np.ndarray, save_path: str = None) -> plt.Figure:
        """
        Create side-by-side comparison figure.
        """
        results = self.evaluate_single(img)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Low-Light Enhancement Comparison', fontsize=16, fontweight='bold')
        
        images = [
            ('Original', img, results['original']),
            ('MSR', results['msr']['image'], results['msr']),
            ('MSRCR', results['msrcr']['image'], results['msrcr']),
            ('CLAHE', results['clahe']['image'], results['clahe'])
        ]
        
        for ax, (title, enhanced, metrics) in zip(axes.flat, images):
            # Convert BGR to RGB for display
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            
            # Format LPIPS (show N/A for original since it compares to itself)
            lpips_val = metrics.get('lpips', 0)
            if title == 'Original':
                lpips_str = "N/A"
            elif lpips_val < 0:
                lpips_str = "N/A"
            else:
                lpips_str = f"{lpips_val:.3f}"
            
            ax.set_title(f'{title}\nNIQE: {metrics["niqe"]:.3f} | LPIPS: {lpips_str} | Time: {metrics["time"]:.2f}ms', 
                        fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison figure saved to {save_path}")
        
        return fig
    
    def print_results(self, summary: Dict):
        """Print formatted results table."""
        print("\n" + "="*80)
        print("LOW-LIGHT ENHANCEMENT EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n{'Method':<12} {'NIQE (↓)':<18} {'LPIPS (↓)':<18} {'Time (ms)':<18} {'Memory (MB)':<18}")
        print("-"*80)
        
        for method in ['original', 'msr', 'msrcr', 'clahe']:
            m = summary[method]
            niqe_str = f"{m['niqe_mean']:.3f} ± {m['niqe_std']:.3f}"
            lpips_str = f"{m['lpips_mean']:.4f} ± {m['lpips_std']:.4f}" if m['lpips_mean'] >= 0 else "N/A"
            time_str = f"{m['time_mean']:.2f} ± {m['time_std']:.2f}"
            mem_str = f"{m['memory_mean']:.2f} ± {m['memory_std']:.2f}"
            
            print(f"{method.upper():<12} {niqe_str:<18} {lpips_str:<18} {time_str:<18} {mem_str:<18}")
        
        print("-"*80)
        print("\nNote: Lower NIQE and LPIPS scores indicate better quality.")
        print("="*80)


# ============================================================================
# MAIN EXECUTION - SEPARATE FUNCTIONS
# ============================================================================

def _get_dataset_path():
    """Find ExDark dataset or create synthetic images."""
    possible_paths = [
        "ExDark",
        "data/ExDark",
        "../ExDark",
        "datasets/ExDark",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def _download_sample_images():
    """Create sample low-light images if no dataset is available."""
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    print("Creating synthetic low-light test images...")
    
    for i in range(5):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        for y in range(480):
            for x in range(640):
                img[y, x] = [
                    int(100 + 50 * np.sin(x / 50)),
                    int(100 + 50 * np.sin(y / 50)),
                    int(100 + 50 * np.sin((x + y) / 70))
                ]
        
        cv2.rectangle(img, (50 + i*20, 50), (200 + i*20, 200), (180, 120, 80), -1)
        cv2.circle(img, (400, 300), 100, (100, 180, 140), -1)
        cv2.ellipse(img, (300, 350), (80, 40), 45, 0, 360, (140, 100, 180), -1)
        
        gamma = 2.0 + i * 0.25
        table = np.array([((j / 255.0) ** gamma) * 255 for j in range(256)]).astype(np.uint8)
        low_light = cv2.LUT(img, table)
        
        cv2.imwrite(str(sample_dir / f"synthetic_lowlight_{i+1}.jpg"), low_light)
        cv2.imwrite(str(sample_dir / f"synthetic_reference_{i+1}.jpg"), img)
    
    print(f"Created {5} synthetic test images in {sample_dir}/")
    return str(sample_dir)


def run_evaluation(num_samples: int = 20, dataset_path: str = None) -> Dict:
    """
    Run quantitative evaluation on dataset samples.
    """
    print("="*80)
    print("LOW-LIGHT IMAGE ENHANCEMENT EVALUATION")
    print("Methods: MSR, MSRCR (Retinex-based), CLAHE")
    print("Metrics: NIQE, LPIPS, Processing Time, Memory Usage")
    print("="*80)
    
    # Find dataset
    if dataset_path is None:
        dataset_path = _get_dataset_path()
    
    if dataset_path is None or not os.path.exists(dataset_path):
        print("\nExDark dataset not found. Will use synthetic images.")
        print("To use ExDark, download from: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset")
        dataset_path = _download_sample_images()
    else:
        print(f"\nUsing dataset at: {dataset_path}")
    
    # Initialize evaluator and run
    evaluator = LowLightEvaluator(dataset_path)
    
    print(f"\nRunning evaluation on {num_samples} samples...")
    summary = evaluator.evaluate_dataset(num_samples)
    
    # Print results
    evaluator.print_results(summary)
    
    return summary


def run_visual_comparison(image_path: str, save_path: str = "lowlight_comparison.png", show: bool = True) -> plt.Figure:
    print("="*80)
    print("LOW-LIGHT ENHANCEMENT VISUAL COMPARISON")
    print("="*80)
    
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    print(f"\nProcessing image: {image_path}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Initialize evaluator (no dataset needed for single image)
    evaluator = LowLightEvaluator(dataset_path=None)
    
    # Create comparison figure
    fig = evaluator.create_comparison_figure(img, save_path=save_path)
    
    if show:
        plt.show()
    
    print(f"\nComparison saved to: {save_path}")
    
    return fig



if __name__ == "__main__":
    # Run quantitative evaluation
    summary = run_evaluation(num_samples=200)

    # Run visual comparison on specific image
    fig = run_visual_comparison("ExDark/Chair/2015_03777.png")

