"""
Some important parameters and setting for GPS simulation
"""


# fundamental frequency
f0 = 10.23e6

# L1 frequency
f1 = 154 * f0

# L2 freqency
f2 = 120 * f0

# C/A chip rate
f_ca = 1.023e6

# carrier aiding
k_ca = f_ca / f1

# C/A code period
T_ca = 1e-3

# C/A code delay for each SV PRN number
delay = [  5,   6,   7,   8,  17,  18, 139, 140, 141, 251, 252, 254, 255, 256, 257, 258,
         469, 470, 471, 472, 473, 474, 509, 512, 513, 514, 515, 516, 859, 860, 861, 862]

# data rate
f_data = 50

# doppler factor
k_data = f_data / f1

# sampling rate, recommended to be a multiple of f_ca for better visualization
fs = f0

# carrier wave intermediate frequency (IF), fi < fs / 2
fi = 4 * f_ca

# sample navigation data broadcast on each satellite
data = [[0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, 1],
        [0, 1, 0, 0, 1, 0]]
