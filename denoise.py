"""
Noise Removal: Non-Local Means (NLM) and BM3D
Evaluation with NIQE, LPIPS metrics and runtime analysis
"""

import os
import cv2
import numpy as np
import time
import tracemalloc
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# For BM3D
try:
    import bm3d
    from bm3d import BM3DProfile
    BM3D_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    BM3D_AVAILABLE = False
    print("Warning: bm3d not installed. BM3D method will be unavailable.")
    print("Install with: pip install bm3d")

# For LPIPS and NIQE (reuse from lowlight.py structure)
# Use importlib to safely check if modules are available without crashing
LPIPS_AVAILABLE = False
try:
    import importlib.util
    # Check if torch can be imported
    torch_spec = importlib.util.find_spec("torch")
    lpips_spec = importlib.util.find_spec("lpips")
    if torch_spec is not None and lpips_spec is not None:
        # Try importing with a timeout-like approach
        import sys
        import io
        # Redirect stderr to catch errors
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            import torch
            import lpips
            LPIPS_AVAILABLE = True
        except Exception:
            LPIPS_AVAILABLE = False
        finally:
            sys.stderr = old_stderr
except Exception:
    LPIPS_AVAILABLE = False

SKIMAGE_AVAILABLE = False
try:
    from skimage.restoration import estimate_sigma
    from skimage import img_as_float
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False
    # Silently fail - we have fallback noise estimation

MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    # Silently fail - visualization will be skipped

# Import NIQE from lowlight.py if available, otherwise define it
try:
    from lowlight import NIQE, LPIPSCalculator
    NIQE_AVAILABLE = True
    LPIPS_CALC_AVAILABLE = True
except ImportError:
    # Define minimal NIQE if not available
    NIQE_AVAILABLE = False
    LPIPS_CALC_AVAILABLE = False
    print("Warning: Could not import NIQE/LPIPS from lowlight.py. Some metrics may be unavailable.")


# ============================================================================
# NOISE GENERATION
# ============================================================================

class NoiseGenerator:
    """
    Generate synthetic noise for testing denoising methods.
    Supports Gaussian and Poisson noise.
    """
    
    @staticmethod
    def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
        """
        Add Gaussian noise to image.
        
        Args:
            img: Input image (uint8, [0, 255])
            sigma: Noise standard deviation (in [0, 255] range)
        
        Returns:
            Noisy image (uint8)
        """
        img_float = img.astype(np.float32)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy = np.clip(img_float + noise, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def add_poisson_noise(img: np.ndarray) -> np.ndarray:
        """
        Add Poisson noise to image (shot noise).
        
        Args:
            img: Input image (uint8)
        
        Returns:
            Noisy image (uint8)
        """
        # Scale to [0, 1] for Poisson
        img_float = img.astype(np.float32) / 255.0
        
        # Generate Poisson noise
        noisy_float = np.random.poisson(img_float * 255.0) / 255.0
        
        # Clip and convert back
        noisy = np.clip(noisy_float * 255.0, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def estimate_noise_sigma(img: np.ndarray) -> float:
        """
        Estimate noise standard deviation from image.
        Uses median absolute deviation (MAD) method.
        
        Args:
            img: Input image
        
        Returns:
            Estimated sigma
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply Gaussian blur to estimate signal
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Compute residual (noise estimate)
        residual = gray.astype(np.float32) - blurred.astype(np.float32)
        
        # MAD estimator: sigma = 1.4826 * median(|residual|)
        mad = np.median(np.abs(residual))
        sigma = 1.4826 * mad
        
        return float(sigma)


# ============================================================================
# METHOD A: NON-LOCAL MEANS (NLM)
# ============================================================================

class NLMDenoiser:
    """
    Non-Local Means Denoising using OpenCV's fastNlMeansDenoising.
    
    NLM Algorithm:
    - For each pixel, find similar patches in a search window
    - Weight patches by similarity (Gaussian-weighted distance)
    - Average similar patches to denoise
    
    Mathematical foundation:
    u^(x) = (1/Z(x)) * sum_{y in Ω} w(x,y) * v(y)
    where w(x,y) = exp(-||P(x) - P(y)||² / h²)
    P(x) is a patch centered at x, h is filtering strength
    """
    
    def __init__(self, h: float = 10.0, 
                 template_window_size: int = 7,
                 search_window_size: int = 21):
        """
        Initialize NLM denoiser.
        
        Args:
            h: Filtering strength (larger = more denoising, but may oversmooth)
            template_window_size: Size of patch used for comparison (typically 5 or 7)
            search_window_size: Size of search region (typically 15, 21, or 31)
        """
        self.h = h
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size
    
    def denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Apply NLM denoising to color image.
        
        Args:
            img: Input noisy image (BGR, uint8)
        
        Returns:
            Denoised image (BGR, uint8)
        """
        if len(img.shape) == 2:
            # Grayscale
            denoised = cv2.fastNlMeansDenoising(
                img,
                h=self.h,
                templateWindowSize=self.template_window_size,
                searchWindowSize=self.search_window_size
            )
        else:
            # Color
            denoised = cv2.fastNlMeansDenoisingColored(
                img,
                h=self.h,
                hColor=self.h * 0.8,  # Slightly less for color channels
                templateWindowSize=self.template_window_size,
                searchWindowSize=self.search_window_size
            )
        
        return denoised
    
    def denoise_grayscale(self, img: np.ndarray) -> np.ndarray:
        """
        Apply NLM denoising to grayscale image.
        
        Args:
            img: Input noisy grayscale image (uint8)
        
        Returns:
            Denoised image (uint8)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        denoised = cv2.fastNlMeansDenoising(
            gray,
            h=self.h,
            templateWindowSize=self.template_window_size,
            searchWindowSize=self.search_window_size
        )
        
        return denoised


# ============================================================================
# METHOD B: BM3D (Block-Matching 3D)
# ============================================================================

class BM3DDenoiser:
    """
    BM3D (Block-Matching 3D) Denoising.
    
    BM3D Algorithm:
    1. Block Matching: Find similar blocks in image
    2. 3D Transform: Stack similar blocks and apply 3D transform
    3. Collaborative Filtering: Hard/soft thresholding in transform domain
    4. Aggregation: Transform back and aggregate results
    
    Mathematical foundation:
    - Groups similar patches into 3D arrays
    - Applies 3D transform (DCT + Haar)
    - Thresholds coefficients: T_hard, T_soft
    - Inverse transform and weighted aggregation
    """
    
    def __init__(self, sigma_psd: float = 25.0, 
                 profile: str = 'np'):
        """
        Initialize BM3D denoiser.
        
        Args:
            sigma_psd: Noise standard deviation (power spectral density)
            profile: Profile string ('np' for normal profile, 'lc' for low complexity, 
                    'vn' for very noisy). Can also use 'normal' or 'high' which map to 'np'
        """
        if not BM3D_AVAILABLE:
            raise ImportError("bm3d library not installed. Install with: pip install bm3d")
        
        self.sigma_psd = sigma_psd
        
        # BM3D uses string profiles: 'np' (normal), 'lc' (low complexity), 'vn' (very noisy)
        # Map user-friendly names to BM3D profile strings
        profile_map = {
            'normal': 'np',
            'high': 'np',
            'np': 'np',
            'lc': 'lc',
            'low': 'lc',
            'vn': 'vn',
            'very_noisy': 'vn'
        }
        
        # Convert to lowercase and map
        profile_lower = profile.lower()
        self.profile = profile_map.get(profile_lower, 'np')
    
    def denoise(self, img: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """
        Apply BM3D denoising to image.
        
        Args:
            img: Input noisy image (BGR or grayscale, uint8)
            sigma: Noise standard deviation. If None, uses self.sigma_psd
        
        Returns:
            Denoised image (same format as input, uint8)
        """
        if sigma is None:
            sigma = self.sigma_psd
        
        # Convert to float [0, 1] for BM3D
        if img.dtype == np.uint8:
            img_float = img.astype(np.float32) / 255.0
        else:
            img_float = img.astype(np.float32)
        
        # BM3D expects sigma in [0, 1] range for normalized images
        # Convert sigma from [0, 255] to [0, 1] range
        sigma_normalized = sigma / 255.0
        
        # Apply BM3D
        # BM3D profile can be string or BM3DProfile enum
        # Try to use BM3DProfile if it's a string that matches an enum value
        try:
            if isinstance(self.profile, str):
                # Try to get BM3DProfile enum if available
                if hasattr(BM3DProfile, self.profile):
                    profile_enum = getattr(BM3DProfile, self.profile)
                else:
                    # Use string directly
                    profile_enum = self.profile
            else:
                profile_enum = self.profile
        except:
            # Fallback to string
            profile_enum = self.profile
        
        if len(img_float.shape) == 2:
            # Grayscale
            denoised_float = bm3d.bm3d(img_float, sigma_psd=sigma_normalized, 
                                      profile=profile_enum)
        else:
            # Color - process each channel separately
            channels = []
            for i in range(img_float.shape[2]):
                channel_denoised = bm3d.bm3d(img_float[:, :, i], 
                                            sigma_psd=sigma_normalized,
                                            profile=profile_enum)
                channels.append(channel_denoised)
            denoised_float = np.stack(channels, axis=2)
        
        # Convert back to uint8
        denoised = np.clip(denoised_float * 255.0, 0, 255).astype(np.uint8)
        
        return denoised
    
    def estimate_and_denoise(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Estimate noise level and apply BM3D denoising.
        
        Args:
            img: Input noisy image
        
        Returns:
            Tuple of (denoised_image, estimated_sigma)
        """
        # Estimate noise
        noise_estimator = NoiseGenerator()
        estimated_sigma = noise_estimator.estimate_noise_sigma(img)
        
        # Use estimated sigma or fallback to default
        if estimated_sigma < 1.0:
            estimated_sigma = self.sigma_psd
        
        # Denoise
        denoised = self.denoise(img, sigma=estimated_sigma)
        
        return denoised, estimated_sigma


# ============================================================================
# BSD DATASET HANDLER (Similar to ExDarkDataset in lowlight.py)
# ============================================================================

class BSDDataset:
    """
    Handler for BSD68/BSD500 datasets with pre-existing noisy images.
    Expected structure:
    - BSD68/original/*.png (clean images)
    - BSD68/noise10/*.png (noisy images with sigma=10)
    - BSD68/noise25/*.png (noisy images with sigma=25)
    - BSD68/noise50/*.png (noisy images with sigma=50)
    """
    
    def __init__(self, root_path: str):
        """
        Initialize BSD dataset.
        
        Args:
            root_path: Path to BSD68 or BSD500 directory
        """
        self.root_path = Path(root_path)
        self.clean_images = {}  # {image_name: path}
        self.noisy_images = {}  # {noise_level: {image_name: path}}
        self._scan_dataset()
    
    def _scan_dataset(self):
        """Scan dataset directory for clean and noisy images."""
        # Look for original/clean images
        original_dir = self.root_path / 'original'
        if original_dir.exists():
            for img_path in original_dir.glob('*.png'):
                if img_path.is_file():
                    self.clean_images[img_path.name] = str(img_path)
            for img_path in original_dir.glob('*.jpg'):
                if img_path.is_file():
                    self.clean_images[img_path.name] = str(img_path)
        
        # Look for noisy images in noise subdirectories
        noise_levels = ['noise10', 'noise25', 'noise50']
        for noise_level in noise_levels:
            noise_dir = self.root_path / noise_level
            if noise_dir.exists():
                self.noisy_images[noise_level] = {}
                for img_path in noise_dir.glob('*.png'):
                    if img_path.is_file():
                        self.noisy_images[noise_level][img_path.name] = str(img_path)
                for img_path in noise_dir.glob('*.jpg'):
                    if img_path.is_file():
                        self.noisy_images[noise_level][img_path.name] = str(img_path)
        
        # If no subdirectories found, try flat structure (fallback)
        # This handles cases like BSD500 where images might be directly in the root
        if not self.clean_images:
            for img_path in self.root_path.glob('*.png'):
                if img_path.is_file():
                    self.clean_images[img_path.name] = str(img_path)
            for img_path in self.root_path.glob('*.jpg'):
                if img_path.is_file():
                    self.clean_images[img_path.name] = str(img_path)
            # Also check common subdirectories like 'images' or 'test'
            for subdir in ['images', 'test', 'val', 'train']:
                subdir_path = self.root_path / subdir
                if subdir_path.exists():
                    for img_path in subdir_path.glob('*.png'):
                        if img_path.is_file():
                            self.clean_images[img_path.name] = str(img_path)
                    for img_path in subdir_path.glob('*.jpg'):
                        if img_path.is_file():
                            self.clean_images[img_path.name] = str(img_path)
        
        print(f"Found {len(self.clean_images)} clean images in BSD dataset at {self.root_path}")
        for noise_level, images in self.noisy_images.items():
            print(f"  {noise_level}: {len(images)} noisy images")
    
    def get_image_pairs(self, noise_levels: List[str] = ['noise10', 'noise25', 'noise50']) -> List[Dict]:
        """
        Get pairs of (clean_image, noisy_image) for evaluation.
        
        Args:
            noise_levels: List of noise level directories to use (e.g., ['noise10', 'noise25', 'noise50'])
        
        Returns:
            List of dicts with 'clean', 'noisy', 'noise_level', 'name' keys
        """
        pairs = []
        
        for noise_level in noise_levels:
            # Extract sigma value from noise level name (e.g., 'noise10' -> 10)
            try:
                sigma = float(noise_level.replace('noise', ''))
            except:
                sigma = 10.0  # Default
            
            if noise_level not in self.noisy_images:
                continue
            
            # Match noisy images with clean images by filename
            for img_name, noisy_path in self.noisy_images[noise_level].items():
                if img_name in self.clean_images:
                    clean_path = self.clean_images[img_name]
                    pairs.append({
                        'clean_path': clean_path,
                        'noisy_path': noisy_path,
                        'noise_level': sigma,
                        'name': img_name
                    })
        
        return pairs
    
    def get_sample_pairs(self, n: int = 100, noise_levels: List[str] = ['noise10', 'noise25', 'noise50']) -> List[Dict]:
        """Get a random sample of image pairs."""
        all_pairs = self.get_image_pairs(noise_levels)
        if len(all_pairs) == 0:
            return []
        
        # Limit to n pairs
        if n >= len(all_pairs):
            selected_pairs = all_pairs
        else:
            indices = np.random.choice(len(all_pairs), n, replace=False)
            selected_pairs = [all_pairs[i] for i in indices]
        
        # Load images
        result = []
        for pair in selected_pairs:
            clean_img = cv2.imread(pair['clean_path'])
            noisy_img = cv2.imread(pair['noisy_path'])
            if clean_img is not None and noisy_img is not None:
                result.append({
                    'clean': clean_img,
                    'noisy': noisy_img,
                    'noise_level': pair['noise_level'],
                    'name': pair['name']
                })
        
        return result


# ============================================================================
# EVALUATION METRICS (Reuse from lowlight.py or define minimal versions)
# ============================================================================

def get_niqe_calculator():
    """Get NIQE calculator, either from lowlight.py or create minimal version."""
    if NIQE_AVAILABLE:
        return NIQE()
    else:
        # Minimal fallback - use simple variance-based metric
        class SimpleNIQE:
            def calculate(self, img):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                # Simple metric: lower variance in smooth regions = better
                blurred = cv2.GaussianBlur(gray, (15, 15), 0)
                diff = gray.astype(np.float32) - blurred.astype(np.float32)
                return float(np.std(diff))
        return SimpleNIQE()

class LPIPSWrapper:
    """
    Wrapper for LPIPS calculator that handles NumPy 2.0/Torch compatibility issues.
    """
    def __init__(self, base_calculator):
        self.base_calc = base_calculator
        if LPIPS_AVAILABLE:
            self.device = base_calculator.device if hasattr(base_calculator, 'device') else torch.device('cpu')
            self.loss_fn = base_calculator.loss_fn if hasattr(base_calculator, 'loss_fn') else None
        else:
            self.device = None
            self.loss_fn = None
    
    def calculate(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate LPIPS with NumPy 2.0 compatibility.
        """
        if self.loss_fn is None:
            return -1.0
        
        try:
            # Convert BGR to RGB and normalize to [-1, 1]
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            # Convert to float32 and normalize
            img1_rgb = img1_rgb.astype(np.float32)
            img2_rgb = img2_rgb.astype(np.float32)
            
            # Normalize to [-1, 1]
            img1_normalized = (img1_rgb / 127.5) - 1.0
            img2_normalized = (img2_rgb / 127.5) - 1.0
            
            # Convert to tensor - use torch.tensor() for NumPy 2.0 compatibility
            # This avoids the "Numpy is not available" error
            if not LPIPS_AVAILABLE:
                return -1.0
                
            try:
                # Try torch.from_numpy first (faster if it works)
                img1_tensor = torch.from_numpy(img1_normalized).permute(2, 0, 1).unsqueeze(0)
                img2_tensor = torch.from_numpy(img2_normalized).permute(2, 0, 1).unsqueeze(0)
            except (RuntimeError, TypeError, AttributeError, ValueError) as e:
                # Fallback: use torch.tensor() which works with NumPy 2.0
                # torch.tensor() creates a copy, avoiding the NumPy compatibility issue
                img1_tensor = torch.tensor(img1_normalized.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                img2_tensor = torch.tensor(img2_normalized.copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            
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
            
            # Convert result to float - handle NumPy 2.0 compatibility
            try:
                result = distance.cpu().numpy()
                return float(result)
            except (RuntimeError, AttributeError, ValueError):
                # Fallback: convert tensor to Python float directly
                return float(distance.cpu().item())
                
        except Exception as e:
            # If LPIPS calculation fails, return -1 to indicate unavailable
            return -1.0


def get_lpips_calculator():
    """Get LPIPS calculator with NumPy 2.0 compatibility fix."""
    if LPIPS_CALC_AVAILABLE and LPIPS_AVAILABLE:
        # Try to use the one from lowlight.py, but wrap it to handle NumPy 2.0 issues
        try:
            base_calc = LPIPSCalculator()
            # Return a wrapper that handles NumPy compatibility
            return LPIPSWrapper(base_calc)
        except Exception as e:
            # If we can't create the base calculator, return None
            return None
    else:
        return None


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

class NoiseRemovalEvaluator:
    """
    Comprehensive evaluator for noise removal methods (NLM and BM3D).
    """
    
    def __init__(self):
        """Initialize evaluator with metrics."""
        self.niqe = get_niqe_calculator()
        self.lpips = get_lpips_calculator()
        
        # Results storage
        self.results = {
            'original': {'niqe': [], 'lpips': [], 'time': [], 'memory': []},  # Clean/original images
            'noisy': {'niqe': [], 'lpips': [], 'time': [], 'memory': []},      # Noisy images (before denoising)
            'nlm': {'niqe': [], 'lpips': [], 'time': [], 'memory': []},
            'bm3d': {'niqe': [], 'lpips': [], 'time': [], 'memory': []}
        }
    
    def measure_performance(self, func, *args) -> Tuple[any, float, float]:
        """
        Measure time and memory usage of a denoising function.
        
        Returns:
            Tuple of (result, time_ms, memory_mb)
        """
        tracemalloc.start()
        start_time = time.perf_counter()
        
        result = func(*args)
        
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        time_ms = (end_time - start_time) * 1000
        memory_mb = peak / (1024 * 1024)
        
        return result, time_ms, memory_mb
    
    def evaluate_single(self, noisy_img: np.ndarray, 
                       clean_img: Optional[np.ndarray] = None,
                       nlm_params: Optional[Dict] = None,
                       bm3d_params: Optional[Dict] = None) -> Dict:
        """
        Evaluate both NLM and BM3D on a single noisy image.
        
        Args:
            noisy_img: Noisy input image
            clean_img: Clean reference (optional, for LPIPS)
            nlm_params: NLM parameters dict (h, template_window_size, search_window_size)
            bm3d_params: BM3D parameters dict (sigma_psd, profile)
        
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        # Default parameters
        if nlm_params is None:
            nlm_params = {'h': 10.0, 'template_window_size': 7, 'search_window_size': 21}
        if bm3d_params is None:
            bm3d_params = {'sigma_psd': 25.0, 'profile': 'normal'}
        
        # Evaluate original/clean image if available
        if clean_img is not None:
            niqe_original = self.niqe.calculate(clean_img)
            lpips_original = 0.0
            if self.lpips:
                try:
                    # LPIPS of clean image to itself should be 0, but we can calculate it
                    # For comparison purposes, we'll set it to 0 (perfect match)
                    lpips_original = 0.0
                except:
                    lpips_original = -1.0
            
            results['original'] = {
                'niqe': niqe_original,
                'lpips': lpips_original,
                'time': 0.0,
                'memory': 0.0,
                'image': clean_img
            }
        else:
            results['original'] = None
        
        # Evaluate noisy image (before denoising)
        niqe_noisy = self.niqe.calculate(noisy_img)
        lpips_noisy = 0.0
        if clean_img is not None and self.lpips:
            try:
                lpips_noisy = self.lpips.calculate(clean_img, noisy_img)
            except (RuntimeError, AttributeError, ImportError):
                lpips_noisy = -1.0
        elif self.lpips:
            # If no clean image, compare noisy to itself (should be 0)
            lpips_noisy = 0.0
        
        results['noisy'] = {
            'niqe': niqe_noisy,
            'lpips': lpips_noisy,
            'time': 0.0,
            'memory': 0.0,
            'image': noisy_img
        }
        
        # Initialize denoisers
        nlm = NLMDenoiser(**nlm_params)
        bm3d_denoiser = BM3DDenoiser(**bm3d_params) if BM3D_AVAILABLE else None
        
        # Evaluate NLM
        nlm_result, nlm_time, nlm_mem = self.measure_performance(nlm.denoise, noisy_img)
        niqe_nlm = self.niqe.calculate(nlm_result)
        lpips_nlm = 0.0
        if clean_img is not None and self.lpips:
            try:
                lpips_nlm = self.lpips.calculate(clean_img, nlm_result)
            except (RuntimeError, AttributeError, ImportError):
                lpips_nlm = -1.0  # Mark as unavailable
        elif self.lpips:
            # Compare to noisy image
            try:
                lpips_nlm = self.lpips.calculate(noisy_img, nlm_result)
            except (RuntimeError, AttributeError, ImportError):
                lpips_nlm = -1.0  # Mark as unavailable
        
        results['nlm'] = {
            'niqe': niqe_nlm,
            'lpips': lpips_nlm,
            'time': nlm_time,
            'memory': nlm_mem,
            'image': nlm_result
        }
        
        # Evaluate BM3D
        if bm3d_denoiser:
            bm3d_result, bm3d_time, bm3d_mem = self.measure_performance(
                bm3d_denoiser.denoise, noisy_img
            )
            niqe_bm3d = self.niqe.calculate(bm3d_result)
            lpips_bm3d = 0.0
            if clean_img is not None and self.lpips:
                try:
                    lpips_bm3d = self.lpips.calculate(clean_img, bm3d_result)
                except (RuntimeError, AttributeError, ImportError):
                    lpips_bm3d = -1.0  # Mark as unavailable
            elif self.lpips:
                try:
                    lpips_bm3d = self.lpips.calculate(noisy_img, bm3d_result)
                except (RuntimeError, AttributeError, ImportError):
                    lpips_bm3d = -1.0  # Mark as unavailable
            
            results['bm3d'] = {
                'niqe': niqe_bm3d,
                'lpips': lpips_bm3d,
                'time': bm3d_time,
                'memory': bm3d_mem,
                'image': bm3d_result
            }
        else:
            results['bm3d'] = None
        
        return results
    
    def evaluate_synthetic_dataset(self, image_pairs: List[Dict]) -> Dict:
        """
        Evaluate on pre-existing noisy images from BSD dataset.
        
        Args:
            image_pairs: List of dicts with 'clean', 'noisy', 'noise_level', 'name' keys
        
        Returns:
            Aggregated results dictionary
        """
        all_results = []
        for pair in image_pairs:
            clean_img = pair['clean']
            noisy_img = pair['noisy']
            sigma = pair['noise_level']
            
            # Evaluate with known noise level
            bm3d_params = {'sigma_psd': sigma, 'profile': 'normal'}
            result = self.evaluate_single(noisy_img, clean_img=clean_img,
                                        bm3d_params=bm3d_params)
            result['noise_level'] = sigma
            result['name'] = pair.get('name', 'unknown')
            all_results.append(result)
            
            # Store in results
            for method in ['original', 'noisy', 'nlm', 'bm3d']:
                if result.get(method):
                    self.results[method]['niqe'].append(result[method]['niqe'])
                    self.results[method]['time'].append(result[method]['time'])
                    self.results[method]['memory'].append(result[method]['memory'])
                    if 'lpips' in result[method] and result[method]['lpips'] >= 0:
                        self.results[method]['lpips'].append(result[method]['lpips'])
        
        return self._aggregate_results()
    
    def evaluate_real_dataset(self, image_paths: List[str],
                             nlm_params: Optional[Dict] = None,
                             bm3d_params: Optional[Dict] = None) -> Dict:
        """
        Evaluate on real noisy images.
        
        Args:
            image_paths: List of paths to noisy images
            nlm_params: NLM parameters
            bm3d_params: BM3D parameters
        
        Returns:
            Aggregated results dictionary
        """
        all_results = []
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            print(f"Processing: {os.path.basename(img_path)}")
            
            # Estimate noise for BM3D
            noise_gen = NoiseGenerator()
            estimated_sigma = noise_gen.estimate_noise_sigma(img)
            
            if bm3d_params is None:
                bm3d_params = {'sigma_psd': estimated_sigma, 'profile': 'normal'}
            elif 'sigma_psd' not in bm3d_params:
                bm3d_params['sigma_psd'] = estimated_sigma
            
            result = self.evaluate_single(img, clean_img=None,
                                        nlm_params=nlm_params,
                                        bm3d_params=bm3d_params)
            result['image_path'] = img_path
            all_results.append(result)
            
            # Store in results (original not available for real noise experiments, but noisy is)
            for method in ['noisy', 'nlm', 'bm3d']:
                if result.get(method):
                    self.results[method]['niqe'].append(result[method]['niqe'])
                    self.results[method]['time'].append(result[method]['time'])
                    self.results[method]['memory'].append(result[method]['memory'])
                    if 'lpips' in result[method] and result[method]['lpips'] >= 0:
                        self.results[method]['lpips'].append(result[method]['lpips'])
        
        return self._aggregate_results()
    
    def _aggregate_results(self) -> Dict:
        """Aggregate results into summary statistics."""
        summary = {}
        
        for method in ['original', 'noisy', 'nlm', 'bm3d']:
            if self.results[method]['niqe']:
                summary[method] = {
                    'niqe_mean': np.mean(self.results[method]['niqe']),
                    'niqe_std': np.std(self.results[method]['niqe']),
                    'lpips_mean': np.mean([x for x in self.results[method]['lpips'] if x >= 0]) if self.results[method]['lpips'] else 0,
                    'lpips_std': np.std([x for x in self.results[method]['lpips'] if x >= 0]) if self.results[method]['lpips'] else 0,
                    'time_mean': np.mean(self.results[method]['time']),
                    'time_std': np.std(self.results[method]['time']),
                    'memory_mean': np.mean(self.results[method]['memory']),
                    'memory_std': np.std(self.results[method]['memory']),
                }
            else:
                summary[method] = {
                    'niqe_mean': 0, 'niqe_std': 0,
                    'lpips_mean': 0, 'lpips_std': 0,
                    'time_mean': 0, 'time_std': 0,
                    'memory_mean': 0, 'memory_std': 0,
                }
        
        return summary
    
    def create_comparison_figure(self, noisy_img: np.ndarray,
                                nlm_result: np.ndarray,
                                bm3d_result: Optional[np.ndarray] = None,
                                clean_img: Optional[np.ndarray] = None,
                                save_path: Optional[str] = None,
                                nlm_metrics: Optional[Dict] = None,
                                bm3d_metrics: Optional[Dict] = None) -> Optional[any]:
        """
        Create side-by-side comparison figure (similar to lowlight.py style).
        
        Args:
            noisy_img: Noisy input image
            nlm_result: NLM denoised result
            bm3d_result: BM3D denoised result (optional)
            clean_img: Clean reference (optional)
            save_path: Path to save figure
            nlm_metrics: Dictionary with NLM metrics (niqe, lpips, time)
            bm3d_metrics: Dictionary with BM3D metrics (niqe, lpips, time)
        
        Returns:
            Matplotlib figure or None
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, skipping visualization.")
            return None
        
        # Calculate metrics if not provided
        if nlm_metrics is None:
            nlm_niqe = self.niqe.calculate(nlm_result)
            nlm_lpips = 0.0
            nlm_time = 0.0
            if self.lpips and clean_img is not None:
                try:
                    nlm_lpips = self.lpips.calculate(clean_img, nlm_result)
                except (RuntimeError, AttributeError, ImportError):
                    nlm_lpips = -1.0  # Mark as unavailable
            nlm_metrics = {'niqe': nlm_niqe, 'lpips': nlm_lpips, 'time': nlm_time}
        
        if bm3d_result is not None:
            if bm3d_metrics is None:
                bm3d_niqe = self.niqe.calculate(bm3d_result)
                bm3d_lpips = 0.0
                bm3d_time = 0.0
                if self.lpips and clean_img is not None:
                    try:
                        bm3d_lpips = self.lpips.calculate(clean_img, bm3d_result)
                    except (RuntimeError, AttributeError, ImportError):
                        bm3d_lpips = -1.0  # Mark as unavailable
                bm3d_metrics = {'niqe': bm3d_niqe, 'lpips': bm3d_lpips, 'time': bm3d_time}
        
        # Calculate noisy metrics
        niqe_noisy = self.niqe.calculate(noisy_img)
        lpips_noisy = 0.0
        if self.lpips and clean_img is not None:
            try:
                lpips_noisy = self.lpips.calculate(clean_img, noisy_img)
            except (RuntimeError, AttributeError, ImportError):
                lpips_noisy = -1.0  # Mark as unavailable
        
        # Determine layout - single row. Include clean if provided.
        if bm3d_result is not None:
            cols = 4 if clean_img is not None else 3
        else:
            cols = 3 if clean_img is not None else 2
        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
        axes = axes if hasattr(axes, '__iter__') else [axes]
        
        fig.suptitle('Noise Removal Comparison: NLM vs BM3D', fontsize=16, fontweight='bold')
        
        idx = 0
        
        # Show clean image if available
        if clean_img is not None:
            rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(rgb)
            clean_niqe = self.niqe.calculate(clean_img)
            axes[idx].set_title(f'Clean (Reference)\nNIQE: {clean_niqe:.3f} | LPIPS: N/A | Time: N/A', 
                              fontsize=10)
            axes[idx].axis('off')
            idx += 1
        
        # Show noisy image
        rgb = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(rgb)
        lpips_str = f"{lpips_noisy:.3f}" if lpips_noisy > 0 else "N/A"
        axes[idx].set_title(f'Noisy\nNIQE: {niqe_noisy:.3f} | LPIPS: {lpips_str} | Time: N/A', 
                          fontsize=10)
        axes[idx].axis('off')
        idx += 1
        
        # Show NLM result
        rgb = cv2.cvtColor(nlm_result, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(rgb)
        lpips_str = f"{nlm_metrics['lpips']:.3f}" if nlm_metrics['lpips'] > 0 else "N/A"
        axes[idx].set_title(f'NLM Denoised\nNIQE: {nlm_metrics["niqe"]:.3f} | LPIPS: {lpips_str} | Time: {nlm_metrics["time"]:.2f}ms', 
                          fontsize=10)
        axes[idx].axis('off')
        idx += 1
        
        # Show BM3D result if available
        if bm3d_result is not None:
            rgb = cv2.cvtColor(bm3d_result, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(rgb)
            lpips_str = f"{bm3d_metrics['lpips']:.3f}" if bm3d_metrics['lpips'] > 0 else "N/A"
            axes[idx].set_title(f'BM3D Denoised\nNIQE: {bm3d_metrics["niqe"]:.3f} | LPIPS: {lpips_str} | Time: {bm3d_metrics["time"]:.2f}ms', 
                              fontsize=10)
            axes[idx].axis('off')
            idx += 1
        
        # Hide unused axes
        for i in range(idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison figure saved to {save_path}")
        
        return fig
    
    def print_results(self, summary: Dict):
        """Print formatted results table."""
        print("\n" + "="*100)
        print("NOISE REMOVAL EVALUATION RESULTS")
        print("="*100)
        
        print(f"\n{'Method':<12} {'NIQE (↓)':<20} {'LPIPS (↓)':<20} {'Time (ms)':<20} {'Memory (MB)':<20}")
        print("-"*100)
        
        # Show original first, then noisy, then NLM and BM3D
        method_order = ['original', 'noisy', 'nlm', 'bm3d']
        for method in method_order:
            if method in summary and len(self.results[method]['niqe']) > 0:
                m = summary[method]
                niqe_str = f"{m['niqe_mean']:.3f} ± {m['niqe_std']:.3f}"
                lpips_str = f"{m['lpips_mean']:.4f} ± {m['lpips_std']:.4f}" if m['lpips_mean'] >= 0 else "N/A"
                
                # For original and noisy, time and memory are always N/A (no processing)
                if method in ['original', 'noisy']:
                    time_str = "N/A"
                    mem_str = "N/A"
                else:
                    time_str = f"{m['time_mean']:.2f} ± {m['time_std']:.2f}"
                    mem_str = f"{m['memory_mean']:.2f} ± {m['memory_std']:.2f}"
                
                method_name = "ORIGINAL" if method == 'original' else "NOISY" if method == 'noisy' else method.upper()
                print(f"{method_name:<12} {niqe_str:<20} {lpips_str:<20} {time_str:<20} {mem_str:<20}")
        
        print("-"*100)
        print("\nNote: Lower NIQE and LPIPS scores indicate better quality.")
        if summary.get('nlm', {}).get('lpips_mean', 0) < 0:
            print("\nNote: LPIPS is unavailable due to NumPy/Torch compatibility issues.")
            print("      This is a known issue with NumPy 2.0 and older PyTorch versions.")
            print("      NIQE, Time, and Memory metrics are still available.")
        print("="*100)


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def _get_bsd_dataset_path():
    """Find BSD68/BSD500 dataset in project root or common locations."""
    # Get current working directory (project root)
    project_root = Path.cwd()
    
    possible_paths = [
        # Check in project root first
        str(project_root / "BSD68"),
        str(project_root / "BSD500"),
        # Then check common locations
        "BSD68",
        "BSD500",
        "data/BSD68",
        "data/BSD500",
        "../BSD68",
        "../BSD500",
        "datasets/BSD68",
        "datasets/BSD500",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Verify it contains images
            path_obj = Path(path)
            has_images = (len(list(path_obj.glob("*.jpg"))) > 0 or 
                         len(list(path_obj.glob("*.png"))) > 0 or
                         len(list(path_obj.glob("images/*.jpg"))) > 0 or
                         len(list(path_obj.glob("images/*.png"))) > 0)
            if has_images:
                return path
    
    return None


def run_evaluation(num_samples: int = 20, dataset_path: str = None,
                   nlm_params: Optional[Dict] = None,
                   bm3d_params: Optional[Dict] = None,
                   use_synthetic: bool = False,
                   noise_levels: List[float] = [10, 25, 50]) -> Dict:
    """
    Run quantitative evaluation on dataset samples (similar to lowlight.py's run_evaluation).
    
    Args:
        num_samples: Number of images to evaluate
        dataset_path: Path to directory containing noisy images (default: testing_data/noise)
                     or BSD dataset for synthetic experiments
        nlm_params: NLM parameters dict (optional)
        bm3d_params: BM3D parameters dict (optional)
        use_synthetic: If True, use BSD dataset with synthetic noise (default: False for real noise)
        noise_levels: Noise sigma values for synthetic experiments [10, 25, 50]
    
    Returns:
        Summary results dictionary
    """
    print("="*80)
    print("NOISE REMOVAL EVALUATION")
    print("Methods: NLM, BM3D")
    print("Metrics: NIQE, LPIPS, Processing Time, Memory Usage")
    print("="*80)
    
    evaluator = NoiseRemovalEvaluator()
    
    # Synthetic noise experiments using BSD dataset
    if use_synthetic:
        # Find BSD dataset
        if dataset_path is None:
            dataset_path = _get_bsd_dataset_path()
        
        if dataset_path is None or not os.path.exists(dataset_path):
            print("\nBSD68/BSD500 dataset not found for synthetic experiments.")
            print("Please provide dataset_path or download BSD68/BSD500 dataset.")
            print("For real noise evaluation, set use_synthetic=False")
            return {}
        else:
            print(f"\nUsing BSD dataset at: {dataset_path}")
        
        # Load BSD dataset
        bsd_dataset = BSDDataset(dataset_path)
        
        if not bsd_dataset.clean_images and not bsd_dataset.noisy_images:
            print(f"No images found in BSD dataset at {dataset_path}")
            return {}
        
        # Map noise_levels to directory names (e.g., [10, 25, 50] -> ['noise10', 'noise25', 'noise50'])
        noise_dirs = [f'noise{int(sigma)}' for sigma in noise_levels]
        
        # Try to get image pairs from pre-existing noisy images first
        image_pairs = bsd_dataset.get_sample_pairs(num_samples, noise_levels=noise_dirs)
        
        # If no pre-existing noisy images, generate synthetic noise from clean images
        if not image_pairs and bsd_dataset.clean_images:
            print(f"\nNo pre-existing noisy images found. Generating synthetic noise from clean images...")
            print(f"Found {len(bsd_dataset.clean_images)} clean images")
            
            # Load clean images
            clean_image_paths = list(bsd_dataset.clean_images.values())
            # Limit to num_samples
            if len(clean_image_paths) > num_samples:
                import random
                clean_image_paths = random.sample(clean_image_paths, num_samples)
            else:
                clean_image_paths = clean_image_paths[:num_samples]
            
            # Generate noisy images for each noise level
            noise_gen = NoiseGenerator()
            image_pairs = []
            
            for clean_path in clean_image_paths:
                clean_img = cv2.imread(clean_path)
                if clean_img is None:
                    continue
                
                img_name = Path(clean_path).name
                
                # Generate noisy images for each noise level
                for sigma in noise_levels:
                    noisy_img = noise_gen.add_gaussian_noise(clean_img, sigma)
                    image_pairs.append({
                        'clean': clean_img.copy(),
                        'noisy': noisy_img,
                        'noise_level': sigma,
                        'name': f"{Path(clean_path).stem}_sigma_{sigma}"
                    })
            
            print(f"\nGenerated {len(image_pairs)} image pairs with synthetic noise")
            print(f"Using synthetic noise levels: {noise_levels}")
        
        elif not image_pairs:
            print("No valid image pairs found in BSD dataset")
            print("Expected structure: BSD68/original/ and BSD68/noise10/, noise25/, noise50/")
            print("Or provide clean images in the dataset directory")
            return {}
        else:
            print(f"\nLoaded {len(image_pairs)} image pairs from BSD dataset")
            print(f"Using pre-existing noisy images with levels: {noise_levels}")
        
        # Run evaluation on image pairs (either pre-existing or synthetically generated)
        summary = evaluator.evaluate_synthetic_dataset(image_pairs)
        
    else:
        # Real noise experiments
        if dataset_path is None:
            # Default to testing_data/noise
            possible_paths = [
                "testing_data/noise",
                "data/noise",
                "../testing_data/noise",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    dataset_path = path
                    break
        
        if dataset_path is None or not os.path.exists(dataset_path):
            print(f"\nNoisy image directory not found.")
            print(f"Please provide dataset_path or place images in testing_data/noise/")
            print("Or use use_synthetic=True for BSD dataset experiments")
            return {}
        else:
            print(f"\nUsing real noisy dataset at: {dataset_path}")
        
        # Get all image files
        dataset_path_obj = Path(dataset_path)
        image_files = sorted(dataset_path_obj.glob("*.jpg")) + sorted(dataset_path_obj.glob("*.png"))
        image_paths = [str(p) for p in image_files]
        
        if not image_paths:
            print(f"No images found in {dataset_path}")
            return {}
        
        # Limit to num_samples
        image_paths = image_paths[:num_samples]
        
        print(f"\nFound {len(image_paths)} images, evaluating {len(image_paths)} samples...")
        
        # Run real noise evaluation
        summary = evaluator.evaluate_real_dataset(image_paths, nlm_params, bm3d_params)
    
    # Print results
    evaluator.print_results(summary)
    
    return summary


def run_synthetic_evaluation(clean_image_paths: List[str],
                            noise_levels: List[float] = [10, 25, 50],
                            noise_type: str = 'gaussian') -> Dict:
    """
    DEPRECATED: This function generates synthetic noise.
    For BSD dataset with pre-existing noisy images, use run_evaluation() with use_synthetic=True.
    
    This function is kept for backward compatibility but is not recommended.
    Use run_evaluation() with BSD dataset that already has noisy images in subdirectories.
    """
    print("="*100)
    print("WARNING: This function generates synthetic noise.")
    print("For BSD dataset with pre-existing noisy images, use:")
    print("  run_evaluation(use_synthetic=True, dataset_path='BSD68')")
    print("="*100)
    
    print("SYNTHETIC NOISE REMOVAL EVALUATION (Generating noise)")
    print(f"Methods: NLM, BM3D")
    print(f"Noise Type: {noise_type}, Levels: {noise_levels}")
    print("="*100)
    
    # Load clean images
    clean_images = []
    for path in clean_image_paths:
        img = cv2.imread(path)
        if img is not None:
            clean_images.append(img)
    
    if not clean_images:
        print("No clean images loaded!")
        return {}
    
    print(f"\nLoaded {len(clean_images)} clean images")
    
    # Generate noisy images (old method - not recommended for BSD)
    noise_gen = NoiseGenerator()
    image_pairs = []
    for clean_img in clean_images:
        for sigma in noise_levels:
            if noise_type == 'gaussian':
                noisy_img = noise_gen.add_gaussian_noise(clean_img, sigma)
            else:
                noisy_img = noise_gen.add_poisson_noise(clean_img)
            image_pairs.append({
                'clean': clean_img,
                'noisy': noisy_img,
                'noise_level': sigma,
                'name': f'generated_sigma_{sigma}'
            })
    
    # Initialize evaluator
    evaluator = NoiseRemovalEvaluator()
    
    # Run evaluation
    print(f"\nRunning evaluation on {len(clean_images)} images with {len(noise_levels)} noise levels...")
    summary = evaluator.evaluate_synthetic_dataset(image_pairs)
    
    # Print results
    evaluator.print_results(summary)
    
    return summary


def run_visual_comparison(image_path: str,
                         save_path: str = "denoise_comparison.png",
                         nlm_params: Optional[Dict] = None,
                         bm3d_params: Optional[Dict] = None,
                         clean_image_path: Optional[str] = None,
                         show: bool = True,
                         max_size: int = 1024) -> Optional[any]:
    """
    Create visual comparison for a single image (similar to lowlight.py's run_visual_comparison).
    
    Args:
        noisy_image_path: Path to noisy image
        save_path: Path to save comparison figure
        nlm_params: NLM parameters
        bm3d_params: BM3D parameters
        clean_image_path: Path to clean reference image (optional)
        show: Whether to display figure
        max_size: Maximum dimension (width or height) for resizing large images (default: 1024)
    
    Returns:
        Matplotlib figure or None
    """
    print("="*80)
    print("NOISE REMOVAL VISUAL COMPARISON")
    print("="*80)
    
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    noisy_img = cv2.imread(image_path)
    if noisy_img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Resize if image is too large
    h, w = noisy_img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        noisy_img = cv2.resize(noisy_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized image from {w}x{h} to {new_w}x{new_h} for faster processing")
    
    print(f"\nProcessing image: {image_path}")
    print(f"Image size: {noisy_img.shape[1]}x{noisy_img.shape[0]}")
    
    # Load clean image if provided
    clean_img = None
    if clean_image_path and os.path.exists(clean_image_path):
        clean_img = cv2.imread(clean_image_path)
        if clean_img is not None:
            print(f"Using clean reference: {clean_image_path}")
            # Resize clean image to match noisy image size
            if clean_img.shape[:2] != noisy_img.shape[:2]:
                clean_img = cv2.resize(clean_img, (noisy_img.shape[1], noisy_img.shape[0]), interpolation=cv2.INTER_AREA)
    
    # Initialize evaluator
    evaluator = NoiseRemovalEvaluator()
    
    # Evaluate (this will compute metrics and denoise)
    if nlm_params is None:
        nlm_params = {'h': 10.0, 'template_window_size': 7, 'search_window_size': 21}
    
    if bm3d_params is None:
        noise_gen = NoiseGenerator()
        estimated_sigma = noise_gen.estimate_noise_sigma(noisy_img)
        bm3d_params = {'sigma_psd': estimated_sigma, 'profile': 'normal'}
    
    result = evaluator.evaluate_single(noisy_img, clean_img=clean_img,
                                     nlm_params=nlm_params,
                                     bm3d_params=bm3d_params)
    
    # Create comparison figure with metrics
    fig = evaluator.create_comparison_figure(
        noisy_img, 
        result['nlm']['image'],
        result['bm3d']['image'] if result.get('bm3d') else None,
        clean_img=clean_img,
        save_path=save_path,
        nlm_metrics=result['nlm'],
        bm3d_metrics=result['bm3d'] if result.get('bm3d') else None
    )
    
    if show and MATPLOTLIB_AVAILABLE:
        plt.show()
    
    print(f"\nComparison saved to: {save_path}")
    
    return fig


def generate_output_samples(image_paths: List[str],
                           output_dir: str = "output_sample/denoise",
                           nlm_params: Optional[Dict] = None,
                           bm3d_params: Optional[Dict] = None,
                           max_images: Optional[int] = None) -> List[str]:
    """
    Generate comparison figures for multiple images and save to output_sample/denoise/
    (Similar to how lowlight.py generates output_sample/lowlight/ images)
    
    Args:
        image_paths: List of paths to noisy images
        output_dir: Output directory for comparison figures
        nlm_params: NLM parameters
        bm3d_params: BM3D parameters
        max_images: Maximum number of images to process (None = all)
    
    Returns:
        List of saved file paths
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    # Limit number of images if specified
    if max_images:
        image_paths = image_paths[:max_images]
    
    print("="*80)
    print(f"GENERATING OUTPUT SAMPLES FOR {len(image_paths)} IMAGES")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    for i, img_path in enumerate(image_paths):
        img_name = Path(img_path).stem
        
        # Generate output filename
        output_filename = f"{img_name}.png"
        save_path = str(output_path / output_filename)
        
        print(f"\n[{i+1}/{len(image_paths)}] Processing: {img_name}")
        
        try:
            # Generate comparison figure
            fig = run_visual_comparison(
                img_path,
                save_path=save_path,
                nlm_params=nlm_params,
                bm3d_params=bm3d_params,
                show=False
            )
            
            if fig is not None:
                saved_paths.append(save_path)
                # Close figure to free memory
                if MATPLOTLIB_AVAILABLE:
                    plt.close(fig)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    
    print("\n" + "="*80)
    print(f"Generated {len(saved_paths)} comparison figures in {output_dir}")
    print("="*80)
    
    return saved_paths


if __name__ == "__main__":
    # Run quantitative evaluation on real noisy images (similar to lowlight.py)
    summary = run_evaluation(num_samples=20)
    
    # For synthetic noise experiments with BSD dataset:
    # summary = run_evaluation(num_samples=20, use_synthetic=True, noise_levels=[10, 25, 50])
    
    # Run visual comparison on specific image (similar to lowlight.py)
    noise_dir = Path("testing_data/noise")
    if noise_dir.exists():
        image_files = sorted(noise_dir.glob("*.jpg")) + sorted(noise_dir.glob("*.png"))
        if image_files:
            run_visual_comparison(str(image_files[0]))

