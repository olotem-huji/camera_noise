import numpy as np
import rawpy
import matplotlib.pyplot as plt
from pathlib import Path

from consts import shutter_speeds
from utils import get_green_channel

# --- CONFIGURATION ---
folder_path = Path(r"C:\Physics\Year 3\Lab\Noise\Data\10.05\Dark")
extension = "*.DNG"
margin = 256  # Resulting in a 512x512 crop
SHOW = False


def calculate_radial_profile(psd_2d):
    """Calculates the azimuthal average of a 2D PSD."""
    h, w = psd_2d.shape
    y, x = np.indices((h, w))
    center_y, center_x = h // 2, w // 2

    # Calculate distance from center for every pixel
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    r = r.astype(np.int32)  # Convert to integer bins

    # Fast averaging using bincount:
    # Sum of PSD values at each radius / Number of pixels at each radius
    tbin = np.bincount(r.ravel(), psd_2d.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / nr

    return radial_profile


files = sorted(list(folder_path.glob(extension)))
num_pairs = len(files) // 2
avg_psd_values = []

print(f"Processing {num_pairs} pairs...")

for i in range(num_pairs):
    img1 = get_green_channel(files[2 * i])
    img2 = get_green_channel(files[2 * i + 1])

    if img1 is None or img2 is None: continue

    h, w = img1.shape
    crop1 = img1[h // 2 - margin: h // 2 + margin, w // 2 - margin: w // 2 + margin]
    crop2 = img2[h // 2 - margin: h // 2 + margin, w // 2 - margin: w // 2 + margin]

    # Difference image normalized to single-frame variance
    diff = (crop1 - crop2) / np.sqrt(2.0)
    diff -= np.mean(diff)  # Remove DC component

    # 2D FFT and Power Spectral Density
    fft_img = np.fft.fft2(diff)
    psd_2d = np.abs(np.fft.fftshift(fft_img)) ** 2 / diff.size

    mean_psd = np.mean(psd_2d)
    avg_psd_values.append(mean_psd)

    # --- Plotting Radial PSD ---
    if i % 5 == 0 and SHOW:
        radial_psd = calculate_radial_profile(psd_2d)

        # Optional: "Heal" the DC dip (index 0 in radial profile) for a cleaner plot
        if len(radial_psd) > 1:
            radial_psd[0] = radial_psd[1]

        # We only plot up to 'margin' (256) because corners don't form full circles
        x_axis = np.arange(margin)

        plt.figure(figsize=(8, 4))
        plt.plot(x_axis, radial_psd[:margin], color='teal', lw=1.5)
        plt.axhline(y=mean_psd, color='orange', linestyle='--', alpha=0.8, label=f'Mean PSD')

        plt.title(f"Pair {i} (Exp: {shutter_speeds[i]:.4f}s) - Radial PSD")
        plt.xlabel('Spatial Frequency (Pixels from Center)')
        plt.ylabel('Power')
        # plt.yscale('log') # Uncomment if you prefer log scale
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

# --- Summary Plot: Avg PSD vs Image Number ---

z = np.polyfit(shutter_speeds, avg_psd_values, 1)
p = np.poly1d(z)

y_pred = p(shutter_speeds)
y_actual = np.array(avg_psd_values)
ss_res = np.sum((y_actual - y_pred) ** 2)
ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print("\n--- Fit Results ---")
print(f"Slope (Gain): {z[0]:.5f}")
print(f"Intercept: {z[1]:.5f}")
print(f"R-squared: {r_squared:.5f}")

plt.figure(figsize=(10, 5))
plt.scatter(shutter_speeds, avg_psd_values, color='darkblue')
plt.plot(shutter_speeds, p(shutter_speeds), "r--", label=f"Fit: y={z[0]:.4f}x + {z[1]:.4f}, R²={r_squared:.3f}")
plt.title('Average Power Spectral Density per Shutter Speed', fontdict={"fontsize": 20})
plt.xlabel('Shutter Speed (s)')
plt.ylabel('Average PSD Value')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(r"C:\Physics\Year 3\Lab\Noise\Camera Noise\Plots\PSD by Shutter Speed.png")
plt.show()