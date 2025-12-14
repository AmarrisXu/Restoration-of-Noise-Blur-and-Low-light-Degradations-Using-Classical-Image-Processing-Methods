import os
import cv2
import numpy as np
import glob

# ============================
# 1. Create output folders
# ============================
input_dir = "253/train2017"
output_clean = "253/blur_dataset/clean"
output_blur = "253/blur_dataset/blurred"
output_kernel = "253/blur_dataset/kernel"

os.makedirs(output_clean, exist_ok=True)
os.makedirs(output_blur, exist_ok=True)
os.makedirs(output_kernel, exist_ok=True)

# ============================
# 2. Motion blur kernel
# ============================
def motion_blur_kernel(length=15, angle=0):
    kernel = np.zeros((length, length))
    center = length // 2
    kernel[center, :] = 1

    # rotate kernel
    rot_mat = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (length, length))

    kernel = kernel / kernel.sum()
    return kernel

# ============================
# 3. Blur settings (RGB)
# ============================
lengths = [7, 15, 25]      # blur strength
angles = [0, 45, 90]       # blur directions

# ============================
# 4. Process all images (RGB)
# ============================
image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
print(f"Found {len(image_paths)} images in train2017")

for img_path in image_paths:
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)

    if img is None:
        print("Error reading:", img_path)
        continue

    # Save clean RGB image
    cv2.imwrite(os.path.join(output_clean, img_name), img)

    # Generate multiple blurred RGB images
    for L in lengths:
        for A in angles:
            kernel = motion_blur_kernel(L, A)

            # apply blur to RGB image
            blurred_rgb = cv2.filter2D(img, -1, kernel)

            # file names
            blur_name = img_name.replace(".jpg", f"_L{L}_A{A}.png")
            kernel_name = img_name.replace(".jpg", f"_L{L}_A{A}.npy")

            # save results
            cv2.imwrite(os.path.join(output_blur, blur_name), blurred_rgb)
            np.save(os.path.join(output_kernel, kernel_name), kernel)

            print("Created:", blur_name)

print("RGB blur dataset generation complete!")
