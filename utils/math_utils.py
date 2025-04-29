import numpy as np

def db_to_linear(db_value):
    """
    convert dB value to linear power
    exmp: 10 dB = 10.0 → 10 ** (10/10) = 10.0
    """
    return 10 ** (db_value / 10.0)

def linear_to_db(linear_value):
    """
    convert linear power value to dB
    """
    if linear_value <= 0:
        raise ValueError("Linear value can't be zero or negative.")
    return 10 * np.log10(linear_value)

def snr(signal_power, noise_power):
    """
    Calculates SNR (Signal-to-Noise Ratio) (for linear values).
    """
    if noise_power == 0:
        raise ZeroDivisionError("Noise power cannot be zero.")
    return signal_power / noise_power

def jnr(jamming_power, noise_power):
    """
    Calculates JNR (Jamming-to-Noise Ratio) 
    """
    return jamming_power / noise_power

def normalize(value, min_val, max_val):
    """
    Normalize a value to a range [0, 1] based on min and max values.
    """
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def calculate_distance(pos1, pos2):
    """Calculates the Euclidean distance between two points."""
    return np.linalg.norm(np.array(pos1) - np.array(pos2))
