import numpy as np
import rawpy
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
folder_path = Path(r"C:\Physics\Year 3\Lab\Noise\Data\10.05\Dark")
extension = "*.DNG"
margin = 256  # Resulting in a 512x512 crop (Power of 2 is faster for FFT)

SHOW = False

shutter_speeds = [
    1/10585, 1/10585, 1/9340, 1/8357, 1/7217, 1/6107, 1/6107, 1/5122, 1/4071, 1/3053,
    1/2010, 1/1005, 1/902, 1/802, 1/703, 1/601, 1/501, 1/401, 1/300, 1/200,
    1/100, 1/90, 1/80, 1/70, 1/60, 1/50, 1/45, 1/35, 1/40, 1/30, 1/30, 1/20,
    0.2, 1/15, 1/10, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.91, 1.1, 1.2, 1.3, 1.4,
    1.8, 2, 2, 4, 6, 8, 10, 15
]


def get_green_channel(filepath):
    try:
        with rawpy.imread(str(filepath)) as raw:
            return raw.raw_image.astype(np.float64)[0::2, 1::2]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


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

    # Calculate Average PSD for the summary plot later
    avg_psd_values.append(np.mean(psd_2d))

    # --- Plotting Individual PSD ---
    # We'll plot a 1D slice through the center of the 2D PSD
    mid = psd_2d.shape[0] // 2
    if i % 5 == 0 and SHOW:
        plt.plot(psd_2d[mid, :], color='teal', lw=1)
        plt.axhline(y=np.mean(psd_2d), color='orange', linestyle='--', alpha=0.8)
        plt.title(f"Pair {i}: Mean PSD = {avg_psd_values[-1]:.2f}")
        plt.yscale('log')
        plt.grid(True, alpha=0.2)
        plt.show()

# --- Summary Plot: Avg PSD vs Image Number ---

# Calculate the linear fit (y = mx + c)
z = np.polyfit(shutter_speeds, avg_psd_values, 1)
p = np.poly1d(z)

# Calculate R squared
y_pred = p(shutter_speeds)
y_actual = np.array(avg_psd_values)
ss_res = np.sum((y_actual - y_pred)**2)
ss_tot = np.sum((y_actual - np.mean(y_actual))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"{r_squared=}")

print(f"Slope (Gain): {z[0]:.5f}")
print(f"Intercept: {z[1]:.5f}")
print(f"R-squared: {r_squared:.5f}")

plt.figure(figsize=(10, 5))
plt.scatter(shutter_speeds, avg_psd_values)
plt.plot(shutter_speeds, p(shutter_speeds), "r--", label=f"Fit: y={z[0]:.4f}x + {z[1]:.4f}, R^2={r_squared:.3f}")
plt.title('Average Power Spectral Density per Shutter Speed')
plt.xlabel('Shutter Speed')
plt.ylabel('Average PSD Value')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()