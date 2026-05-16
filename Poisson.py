import os
import rawpy
import numpy as np
import matplotlib.pyplot as plt

# Your directory
directory = r"C:\Physics\Year 3\Lab\Noise\Data\10.05\Dark"

# Selecting a range of files from your 'Light' set to show increasing intensity
# I chose these based on your list to get a good spread of signal levels
target_files = [
    "IMG_20260510_131759.dng",
    "IMG_20260510_132007.dng",
    "IMG_20260510_132221.dng",
    "IMG_20260510_132445.dng",
    "IMG_20260510_132710.dng",
    "IMG_20260510_132956.dng"
]

plt.figure(figsize=(15, 10))

for i, filename in enumerate(target_files):
    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
        continue

    try:
        with rawpy.imread(file_path) as raw:
            # 1. Extract Green channel raw data
            # We use a 400x400 central crop to avoid vignetting/edge artifacts
            raw_data = raw.raw_image.astype(np.float64)
            h, w = raw_data.shape
            crop = raw_data[h // 2 - 200:h // 2 + 200, w // 2 - 200:w // 2 + 200]

            # Bayer Green 1 is usually [0::2, 1::2]
            green_pixels = crop[0::2, 1::2].flatten()

            # 2. Subtract black level (Offset)
            black_level = raw.black_level_per_channel[1]
            signal = green_pixels - black_level

            # Statistics
            mean_mu = np.mean(signal)
            var_sigma2 = np.var(signal)

            # 3. Create Subplot
            plt.subplot(2, 3, i + 1)

            # Histogram of actual data
            count, bins, ignored = plt.hist(signal, bins=60, color='green', alpha=0.6, density=True)

            # 4. Overlap Theoretical Gaussian (The limit of Poisson for large N)
            # Normal distribution: (1/sigma*sqrt(2pi)) * exp(-(x-mu)^2 / 2sigma^2)
            x = np.linspace(min(signal), max(signal), 100)
            gaussian = (1 / np.sqrt(2 * np.pi * var_sigma2)) * np.exp(-(x - mean_mu) ** 2 / (2 * var_sigma2))
            plt.plot(x, gaussian, 'r-', lw=2, label='Normal Fit')

            plt.title(f"$\mu$={mean_mu:.1f}, $\sigma^2$={var_sigma2:.1f}")
            plt.xlabel("Intensity (DN)")
            plt.ylabel("Probability Density")
            if i == 0: plt.legend()

    except Exception as e:
        print(f"Could not process {filename}: {e}")

plt.suptitle("Shot Noise Distributions: Evolution of Pixel Intensity Histograms", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()