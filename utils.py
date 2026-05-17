import rawpy
import numpy as np


def get_green_channel(filepath):
    try:
        with rawpy.imread(str(filepath)) as raw:
            g1_black_level = raw.black_level_per_channel[1]
            g1_channel = raw.raw_image.astype(np.float64)[0::2, 1::2]
            corrected_g1_channel = g1_channel - g1_black_level
            return corrected_g1_channel
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
