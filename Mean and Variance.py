import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress

from utils import get_green_channel

plt.style.use("physrev.mplstyle")

# --- CONFIGURATION ---
folder_path = Path(r"C:\Physics\Year 3\Lab\Noise\Data\10.05\Dark")
extension = "*.DNG"
margin = 500  # Size of the central crop
# ---------------------

# 1. Setup Data Collection
files = sorted(list(folder_path.glob(extension)))
num_pairs = len(files) // 2

if num_pairs < 1:
    print(f"Error: Found {len(files)} files. Need at least 2.")
    exit()

means = []
variances = []
max_pixel_values = []

print(f"Processing {num_pairs} pairs for Green1 channel...")

# 2. Main Analysis Loop
for i in range(0, len(files) - 1, 2):
    img1 = get_green_channel(files[i])
    img2 = get_green_channel(files[i + 1])

    if img1 is None or img2 is None:
        continue

    max_pixel_values.append(max(np.max(img1), np.max(img2)))

    h, w = img1.shape
    crop1 = img1[h // 2 - margin: h // 2 + margin, w // 2 - margin: w // 2 + margin]
    crop2 = img2[h // 2 - margin: h // 2 + margin, w // 2 - margin: w // 2 + margin]

    # Calculate Mean Signal and Variance
    means.append(np.mean((crop1 + crop2) / 2.0))
    variances.append(np.var(crop1 - crop2) / 2.0)

means = np.array(means)
variances = np.array(variances)

# 3. Curve Fitting (Linear Relation on ALL data)
slope, intercept, r_value, p_value, std_err = linregress(means, variances)
r_squared = r_value ** 2

# 4. Results Table
print("\n" + "=" * 45)
print(f"               RESULTS: GREEN 1")
print("=" * 45)
print(f"Conversion Gain (k): {slope:.5f} DN/e-")
print(f"System Gain (1/k):   {1 / slope:.2f} e-/DN")
print(f"Read Noise Floor:    {np.sqrt(max(0, intercept)):.2f} DN")
print(f"R-squared (Fit):     {r_squared:.5f}")
print("=" * 45)

# 5. Graph 1: Photon Transfer Curve
plt.figure(figsize=(10, 6))

# Plot the raw data points
plt.scatter(means, variances, alpha=0.6, s=40)

# Plot the linear fit across all points
fit_equation = f'Linear Fit: $y = {slope:.4f}x + {intercept:.2f}$\n($R^2={r_squared:.4f}$)'
plt.plot(means, slope * means + intercept,  "r--", label=fit_equation)

plt.title('Photon Transfer Curve (Green1 Channel)', fontdict={"fontsize": 20})
plt.xlabel('Mean Signal (${DN}$)')
plt.ylabel('Variance (${DN^2}$)')
plt.legend(loc='upper left', framealpha=0.9)
# plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(r"C:\Physics\Year 3\Lab\Noise\Camera Noise\Plots\Mean and Variance.png")
plt.show()
