import numpy as np
import matplotlib.pyplot as plt
from utils import get_green_channel

plt.style.use("physrev.mplstyle")

# Load the images
im1 = get_green_channel(r"C:\Physics\Year 3\Lab\Noise\Data\10.05\Dark\IMG_20260510_131817.dng")
im2 = get_green_channel(r"C:\Physics\Year 3\Lab\Noise\Data\10.05\Dark\IMG_20260510_131818.dng")

# Shift the baseline to 0.
# (If your get_green_channel function already subtracts 64, remove this step)
im1_centered = im1 - 1
im2_centered = im2 - 1

# Exaggerate the noise: Condition -> If pixel is NOT 0, set to 255. Otherwise, 0.
im1_exag = np.where(im1_centered > 0, 255, 0)
im2_exag = np.where(im2_centered > 0, 255, 0)


# --- Figure 0: Original Unprocessed Images ---
fig_orig, axes_orig = plt.subplots(1, 2, figsize=(14, 7))

# Plot Original Image 1
axes_orig[0].imshow(im1, cmap='viridis', vmin=np.min(im1), vmax=np.max(im1))
axes_orig[0].set_title('Image 1: Original Raw Green Channel')
axes_orig[0].axis('off')

# Plot Original Image 2
axes_orig[1].imshow(im2, cmap='viridis', vmin=np.min(im2), vmax=np.max(im2))
axes_orig[1].set_title('Image 2: Original Raw Green Channel')
axes_orig[1].axis('off')

# Explicitly anchor the layout spacing to enforce identical sizing on your slides
fig_orig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90, wspace=0.10)
plt.savefig(r"C:\Physics\Year 3\Lab\Noise\Camera Noise\Plots\Original Images.png")
plt.show()


# --- Figure 1: Exaggerated Noise ---
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Plot Image 1
axes[0].imshow(im1_exag, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Image 1: Exaggerated Noise')
axes[0].axis('off')

# Plot Image 2
axes[1].imshow(im2_exag, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Image 2: Exaggerated Noise')
axes[1].axis('off')

# Explicitly anchor the layout spacing identically
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90, wspace=0.10)
plt.savefig(r"C:\Physics\Year 3\Lab\Noise\Camera Noise\Plots\Exaggerated Noise Images.png")
plt.show()


# --- Figure 2: Zoomed in on the Center (Proportions Preserved) ---
h, w = im1.shape
aspect_ratio = w / h

# Set the vertical zoom margin, then dynamically scale the horizontal margin
zoom_margin_y = 50
zoom_margin_x = int(zoom_margin_y * aspect_ratio)

# Slice the exaggerated arrays using the proportional bounding box
im1_zoom = im1_exag[h//2 - zoom_margin_y: h//2 + zoom_margin_y, w//2 - zoom_margin_x: w//2 + zoom_margin_x]
im2_zoom = im2_exag[h//2 - zoom_margin_y: h//2 + zoom_margin_y, w//2 - zoom_margin_x: w//2 + zoom_margin_x]

fig_zoom, axes_zoom = plt.subplots(1, 2, figsize=(14, 7))

# Plot Zoomed Image 1
axes_zoom[0].imshow(im1_zoom, cmap='gray', vmin=0, vmax=255, interpolation='none')
axes_zoom[0].set_title(f'Image 1: Center Zoom ({im1_zoom.shape[1]}x{im1_zoom.shape[0]})')
axes_zoom[0].axis('off')

# Plot Zoomed Image 2
axes_zoom[1].imshow(im2_zoom, cmap='gray', vmin=0, vmax=255, interpolation='none')
axes_zoom[1].set_title(f'Image 2: Center Zoom ({im2_zoom.shape[1]}x{im2_zoom.shape[0]})')
axes_zoom[1].axis('off')

fig_zoom.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90, wspace=0.10)
plt.savefig(r"C:\Physics\Year 3\Lab\Noise\Camera Noise\Plots\Exaggerated Noise Images Zoom.png")
plt.show()