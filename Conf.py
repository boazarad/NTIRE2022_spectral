# Global Constants
import numpy as np


# Attempt to load private configuration file with confidential competition parameters
try:
    from PrivateConf import *
except ModuleNotFoundError:
    # Sample parameters which are similar to the confidential competition parameters BUT NOT IDENTICAL!
    NOISE = 750
    JPEG_QUALITY = 65

RGB_FILTER_CSV = 'resources/RGB_Camera_QE.csv'
MOSAIC_FILTER_CSV = 'resources/MS_Camera_QE.csv'
ANALOG_CHANNEL_GAIN = np.array([2.2933984, 1, 1.62308182])

TYPICAL_SCENE_REFLECTIVITY = 0.18
MAX_VAL_8_BIT = (2 ** 8 - 1)
MAX_VAL_12_BIT = (2 ** 12 - 1)

SIZE = 512
QUARTER = SIZE // 4
CROP = np.s_[QUARTER:-QUARTER, QUARTER:-QUARTER]  # keep only the center 50% of the image

SUBMISSION_SIZE_LIMIT = 5*10**8  # (500MB)
