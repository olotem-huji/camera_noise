import numpy as np
import rawpy
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress

# --- CONFIGURATION ---
folder_path = Path(r"C:\Physics\Year 3\Lab\Noise\Data\10.05\Dark")
extension = "*.DNG"
margin = 500  # Size of the central crop


# ---------------------

def get_green_channel(filepath):
    """Extracts specifically the Green1 channel from a raw file."""
    try:
        with rawpy.imread(str(filepath)) as raw:
            raw_data = raw.raw_image.astype(np.float64)
            # Slicing for Green1 in a standard RGGB Bayer pattern
            return raw_data[0::2, 1::2]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


# 1. Setup Data Collection
files = sorted(list(folder_path.glob(extension)))
if len(files) < 2:
    print(f"Error: Found {len(files)} files. Need at least 2.")
    exit()

means = []
variances = []

print(f"Processing {len(files) // 2} pairs for Green1 channel...")

# 2. Main Analysis Loop
for i in range(0, len(files) - 1, 2):
    img1 = get_green_channel(files[i])
    img2 = get_green_channel(files[i + 1])

    if img1 is None or img2 is None: continue

    h, w = img1.shape
    crop1 = img1[h // 2 - margin: h // 2 + margin, w // 2 - margin: w // 2 + margin]
    crop2 = img2[h // 2 - margin: h // 2 + margin, w // 2 - margin: w // 2 + margin]

    # Calculate Mean Signal and Variance
    means.append(np.mean((crop1 + crop2) / 2.0))
    variances.append(np.var(crop1 - crop2) / 2.0)

means = np.array(means)
variances = np.array(variances)

# 3. Curve Fitting (Linear Region)
max_idx = np.argmax(variances)
clean_m, clean_v = means[:max_idx + 1], variances[:max_idx + 1]
slope, intercept, r_val, p_val, std_err = linregress(clean_m, clean_v)

# 4. Results Table
print("\n" + "=" * 40)
print(f"CHANNEL: GREEN 1")
print(f"Conversion Gain (k): {slope:.5f} DN/e-")
print(f"System Gain (1/k):   {1 / slope:.2f} e-/DN")
print(f"Read Noise Floor:    {np.sqrt(max(0, intercept)):.2f} DN")
print(f"R-squared (Fit):     {r_val ** 2:.5f}")
print("=" * 40)

# 5. Graph 1: Photon Transfer Curve
plt.figure(figsize=(10, 5))
plt.scatter(means, variances, color='green', alpha=0.5, label='Data Points')
plt.plot(clean_m, slope * clean_m + intercept, color='red', label=f'Linear Fit (R²={r_val ** 2:.3f})')
plt.axvline(x=means[max_idx], color='black', linestyle='--', label='Saturation')
plt.title('Photon Transfer Curve (Green1 Channel)')
plt.xlabel('Mean Signal (DN)')
plt.ylabel('Variance (DN²)')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

# # 6. Graph 2: Noise Probability Distribution
# # Pick a middle exposure pair to show a clear Gaussian distribution
# pair_idx = (len(files) // 2) & ~1
# f1, f2 = files[pair_idx], files[pair_idx + 1]
# c1 = get_green_channel(f1)[h // 2 - 200: h // 2 + 200, w // 2 - 200: w // 2 + 200]
# c2 = get_green_channel(f2)[h // 2 - 200: h // 2 + 200, w // 2 - 200: w // 2 + 200]
#
# noise_data = (c1 - c2).flatten() / np.sqrt(2)
#
# plt.figure(figsize=(10, 5))
# plt.hist(noise_data, bins=80, density=True, alpha=0.6, color='green', edgecolor='darkgreen')
# mu, sigma = np.mean(noise_data), np.std(noise_data)
# x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
# plt.plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)),
#          color='red', lw=2, label=f'Gaussian Fit (σ={sigma:.2f})')
# plt.title(f'Noise Distribution at {np.mean(c1):.1f} DN')
# plt.xlabel('Noise Magnitude (DN)')
# plt.ylabel('Probability Density')
# plt.legend()
# plt.show()