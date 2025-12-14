import os
import glob
import numpy as np
import cv2
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import gaussian_filter
from scipy.special import gamma
import pandas as pd



def aggd_features(imdata):
    imdata = imdata.flatten()
    imdata2 = imdata * imdata

    left_data = imdata[imdata < 0]
    right_data = imdata[imdata >= 0]

    left_mean_sqrt = np.sqrt((left_data*left_data).mean()) if len(left_data)>0 else 0
    right_mean_sqrt = np.sqrt((right_data*right_data).mean()) if len(right_data)>0 else 0

    gamma_hat = left_mean_sqrt / right_mean_sqrt if right_mean_sqrt != 0 else 0

    imdata_sq = imdata2.mean()
    imdata_abs = np.abs(imdata).mean()

    rhat = imdata_sq / (imdata_abs*imdata_abs)
    rhatnorm = rhat * ((gamma(2/gamma_hat)*gamma(2/gamma_hat)) /
                       (gamma(1/gamma_hat)*gamma(3/gamma_hat)))

    return [rhatnorm, left_mean_sqrt, right_mean_sqrt, gamma_hat]


def compute_niqe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = gray / 255.0

    mu = gaussian_filter(gray, 7/6)
    sigma = np.sqrt(gaussian_filter(gray*gray, 7/6) - mu*mu)

    structdis = (gray - mu) / (sigma + 1e-6)

    features = aggd_features(structdis)
    return float(np.sum(features))



def compute_psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2, data_range=255)

def compute_ssim(img1, img2):
    ssim_total = 0
    for c in range(3):
        ssim_total += structural_similarity(img1[:,:,c], img2[:,:,c], data_range=255)
    return ssim_total / 3.0

# LPIPS (CPU version)
lpips_model = lpips.LPIPS(net="vgg").cpu()

def compute_lpips(img1, img2):
    t1 = torch.tensor(img1/255.0, dtype=torch.float32).permute(2,0,1).unsqueeze(0).cpu()
    t2 = torch.tensor(img2/255.0, dtype=torch.float32).permute(2,0,1).unsqueeze(0).cpu()
    d = lpips_model(t1, t2)
    return float(d.item())




clean_dir = "253/blur_dataset/clean"
wiener_dir = "253/deblurred/wiener"
rl_dir = "253/deblurred/rl"

csv_wiener = "253/metrics_wiener.csv"
csv_rl = "253/metrics_rl.csv"

results_wiener = []
results_rl = []



clean_images = glob.glob(os.path.join(clean_dir, "*.jpg"))
print(f"Found {len(clean_images)} clean images.")

for clean_path in clean_images:
    base = os.path.basename(clean_path).replace(".jpg", "")
    clean = cv2.imread(clean_path)

    if clean is None:
        print(f"[ERROR] Cannot read clean image: {clean_path}")
        continue

    matched_blurs = glob.glob(os.path.join(wiener_dir, base + "_*.png"))

    for blur_path in matched_blurs:
        suffix = blur_path.split(base)[1].replace(".png", "")

        w_path = os.path.join(wiener_dir, base + suffix + ".png")
        r_path = os.path.join(rl_dir, base + suffix + ".png")

        w_img = cv2.imread(w_path)
        r_img = cv2.imread(r_path)

        if w_img is None or r_img is None:
            print(f"[WARNING] Missing deblurred images for: {base}{suffix}")
            continue

        if clean.shape != w_img.shape:
            print(f"[WARNING] Shape mismatch: {base}{suffix}")
            continue

        # ======================== Wiener ========================
        psnr_w = compute_psnr(clean, w_img)
        ssim_w = compute_ssim(clean, w_img)
        niqe_w = compute_niqe(w_img)
        lpips_w = compute_lpips(clean, w_img)

        results_wiener.append([base, suffix, psnr_w, ssim_w, niqe_w, lpips_w])

        # ======================== RL ============================
        psnr_r = compute_psnr(clean, r_img)
        ssim_r = compute_ssim(clean, r_img)
        niqe_r = compute_niqe(r_img)
        lpips_r = compute_lpips(clean, r_img)

        results_rl.append([base, suffix, psnr_r, ssim_r, niqe_r, lpips_r])

        print(f"Done: {base}{suffix}")



df_wiener = pd.DataFrame(results_wiener, columns=["name","suffix","psnr","ssim","niqe","lpips"])
df_rl = pd.DataFrame(results_rl, columns=["name","suffix","psnr","ssim","niqe","lpips"])

df_wiener.to_csv(csv_wiener, index=False)
df_rl.to_csv(csv_rl, index=False)

print("\n=== Evaluation Complete ===")
print(f"Wiener results saved to: {csv_wiener}")
print(f"RL results saved to:     {csv_rl}")
