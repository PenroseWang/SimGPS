"""
Utilities for GPS signal structure
"""
import numpy as np


# circular correlation
def circular_correlation(data1, data2):
    return np.correlate(data1, np.hstack((data2[1:], data2)), mode='valid') / len(data1)


