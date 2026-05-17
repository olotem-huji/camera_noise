import numpy as np
import rawpy
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import re
from datetime import datetime
import zoneinfo

from utils import get_green_channel

# --- CONFIGURATION ---
folder_path = Path(r"C:\Physics\Year 3\Lab\Noise\Data\12.05\Thermal")
csv_path = Path(r"C:\Physics\Year 3\Lab\Noise\Data\12.05\Thermal\BatteryTemp.csv")
extension = "*.DNG"
margin = 256  # Resulting in a 512x512 crop
SHOW = False

# Set the correct local timezone
israel_tz = zoneinfo.ZoneInfo("Asia/Jerusalem")

# 1. Load the Battery Temperature Data
temp_df = pd.read_csv(csv_path)
temp_timestamps = temp_df['time'].values / 1e9  # Convert ns to standard Unix seconds
temperatures = temp_df['temperature'].values

# 2. Setup Data Collection
files = sorted(list(folder_path.glob(extension)))
num_pairs = len(files) // 2
avg_psd_values = []
pair_temperatures = []

print(f"Processing {num_pairs} pairs...")

for i in range(num_pairs):
    file1 = files[2 * i]
    file2 = files[2 * i + 1]

    img1 = get_green_channel(file1)
    img2 = get_green_channel(file2)

    if img1 is None or img2 is None: continue

    # --- Extract Time from Filename ---
    # Looks for the standard pattern: 8 digits, underscore, 6 digits (e.g., 20260510_131817)
    match = re.search(r"(\d{8}_\d{6})", file1.name)
    if not match:
        print(f"Warning: Could not extract date from {file1.name}. Skipping.")
        continue

    date_str = match.group(1)

    # Parse the string into a naive datetime object
    local_dt = datetime.strptime(date_str, "%Y%m%d_%H%M%S")

    # Make it timezone-aware (Israel Local Time)
    aware_dt = local_dt.replace(tzinfo=israel_tz)

    # Convert to standard UTC Unix timestamp to match the CSV
    file_timestamp = aware_dt.timestamp()

    # Interpolate to find the exact temperature at that second
    current_temp = np.interp(file_timestamp, temp_timestamps, temperatures)
    pair_temperatures.append(current_temp)

    h, w = img1.shape
    crop1 = img1[h // 2 - margin: h // 2 + margin, w // 2 - margin: w // 2 + margin]
    crop2 = img2[h // 2 - margin: h // 2 + margin, w // 2 - margin: w // 2 + margin]

    # Difference image normalized to single-frame variance
    diff = (crop1 - crop2) / np.sqrt(2.0)
    diff -= np.mean(diff)  # Remove DC component

    # 2D FFT and Power Spectral Density
    fft_img = np.fft.fft2(diff)
    psd_2d = np.abs(np.fft.fftshift(fft_img)) ** 2 / diff.size

    avg_psd_values.append(np.mean(psd_2d))

    # --- Plotting Individual PSD ---
    mid = psd_2d.shape[0] // 2
    if i % 5 == 0 and SHOW:
        plt.plot(psd_2d[mid, :], color='teal', lw=1)
        plt.axhline(y=np.mean(psd_2d), color='orange', linestyle='--', alpha=0.8)
        plt.title(f"Pair {i}: Mean PSD = {avg_psd_values[-1]:.2f} @ {current_temp:.1f}°C")
        plt.yscale('log')
        plt.grid(True, alpha=0.2)
        plt.show()

# --- Plotting Summary ---
plt.figure(figsize=(10, 5))

# Use the explicitly matched temperatures for the X axis
plt.scatter(pair_temperatures, avg_psd_values, color='firebrick', alpha=0.7, edgecolor='black')

plt.title('Average Power Spectral Density by Battery Temperature, ISO=600')
plt.xlabel('Battery Temperature [°C]')
plt.ylabel('Average PSD Value')
plt.grid(True, alpha=0.3, linestyle='--')

# Save and Show
plt.tight_layout()
plt.savefig(r"C:\Physics\Year 3\Lab\Noise\Camera Noise\Plots\PSD by Temp.png")
plt.show()