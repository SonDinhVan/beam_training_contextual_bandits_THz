"""
This file contains the constants and parameters of the system.
"""
# Constants
# Speed of light
LIGHT_SPEED = 3.0 * 10**8
K_B = 1.381 * 10**-23
T0 = 290
NOISE_FIGURE = 7.2

# Channel constants
# Main frequency in GHz
FREQUENCY = 60.0
# Bandwidth GHz
BANDWIDTH = 0.005
# Number of subcarriers
NUM_SUB_CARRIERS = 1
# Noise power in watt for each sub channel
NOISE_POWER = (
    BANDWIDTH / NUM_SUB_CARRIERS * 10**9 * K_B * T0 * 10 ** (NOISE_FIGURE / 10)
)

# Symbol interval
SYMBOL_INTERVAL = 1 / 3.0 / 10**6
# Coherence interval (#symbol)
TAU_C = 10000

# BS Transmit power in Watts
P_BS = 10.0
