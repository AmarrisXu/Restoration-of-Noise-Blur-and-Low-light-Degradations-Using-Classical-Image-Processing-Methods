import os
import glob
import numpy as np
import cv2

# input dirs
blur_dir = "253/blur_dataset/blurred"
kernel_dir = "253/blur_dataset/kernel"

# output dirs
out_wiener = "253/deblurred/wiener"
out_rl = "253/deblurred/rl"
os.makedirs(out_wiener, exist_ok=True)
os.makedirs(out_rl, exist_ok=True)

# =============== insert Wiener and RL functions ===============
# (copy the functions from previous sections here)
def wiener_deblur_gray(blurred, kernel, K=0.01):
    blurred = blurred.astype(np.float32) / 255.0
    kernel = kernel.astype(np.float32)
    H = np.fft.fft2(kernel, s=blurred.shape)
    G = np.fft.fft2(blurred)
    H_conj = np.conj(H)
    F_hat = (H_conj / (np.abs(H)**2 + K)) * G
    f = np.abs(np.fft.ifft2(F_hat))
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)

def wiener_deblur_rgb(img, kernel, K=0.01):
    channels = cv2.split(img)
    return cv2.merge([wiener_deblur_gray(c, kernel, K) for c in channels])

def richardson_lucy_gray(blurred, kernel, iterations=30):
    blurred = blurred.astype(np.float32) / 255.0
    kernel = kernel.astype(np.float32)
    kernel /= kernel.sum()

    estimate = np.full_like(blurred, 0.5)
    kernel_flip = np.flipud(np.fliplr(kernel))

    for _ in range(iterations):
        conv = cv2.filter2D(estimate, -1, kernel)
        relative_blur = blurred / (conv + 1e-6)
        estimate *= cv2.filter2D(relative_blur, -1, kernel_flip)

    return (np.clip(estimate, 0, 1) * 255).astype(np.uint8)

def rl_deblur_rgb(img, kernel, iterations=30):
    channels = cv2.split(img)
    return cv2.merge([richardson_lucy_gray(c, kernel, iterations) for c in channels])

# ===================== Batch Processing =======================
blur_list = sorted(glob.glob(os.path.join(blur_dir, "*.png")))

print(f"Found {len(blur_list)} blurred images")

for blur_path in blur_list:
    img_name = os.path.basename(blur_path)
    # identify kernel name
    kernel_name = img_name.replace(".png", ".npy")

    blurred = cv2.imread(blur_path)   # RGB
    kernel = np.load(os.path.join(kernel_dir, kernel_name))

    # Wiener (RGB)
    wiener_img = wiener_deblur_rgb(blurred, kernel)
    cv2.imwrite(os.path.join(out_wiener, img_name), wiener_img)

    # RL (RGB)
    rl_img = rl_deblur_rgb(blurred, kernel, iterations=30)
    cv2.imwrite(os.path.join(out_rl, img_name), rl_img)

    print("Processed:", img_name)

print("All deblurring completed!")
