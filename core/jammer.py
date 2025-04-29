"""
Defines the Jammer class representing a jammer entity in the simulation.

Each Jammer instance holds information about its position, transmission 
properties (gain, loss, bandwidth), and current power output.
"""
from utils.math_utils import db_to_linear
import numpy as np

class Jammer:
    """
    Represents a single jammer agent in the electromagnetic environment.

    Attributes:
        power (float): Current transmission power in Watts. Set dynamically during simulation step.
        gj (float): Jammer antenna gain (linear units).
        loss (float): Jammer transmission loss factor (linear units).
        latm (float): Atmospheric loss factor (linear units).
        bj (float): Jammer signal bandwidth (Hz, linear units).
        position (np.ndarray): Numpy array representing the (x, y) position.
        power_max (float): Maximum possible transmission power (Watts). Set after initialization.
        power_min (float): Minimum possible transmission power (Watts). Set after initialization.
    """
    def __init__(self, power, gj, loss, latm, bj, position):
        """
        Initializes a Jammer instance.

        Note: Assumes input parameters gj, loss, latm are in dB and converts them to linear.
              Assumes input bj is already in linear units (Hz).
              Initial 'power' is typically set to 0 and updated later.
              power_max and power_min are set externally after initialization.

        Args:
            power (float): Initial transmission power (usually 0).
            gj (float): Jammer antenna gain (in dB).
            loss (float): Jammer transmission loss factor (in dB).
            latm (float): Atmospheric loss factor (in dB).
            bj (float): Jammer signal bandwidth (in Hz).
            position (list or np.ndarray): (x, y) coordinates.
        """
        # Store the current power (this will be updated by agent actions)
        self.power = power 
        # Convert dB values to linear factors for calculations
        self.gj = db_to_linear(gj)
        self.loss = db_to_linear(loss)
        self.latm = db_to_linear(latm)
        # Bandwidth is assumed to be provided in linear Hz
        self.bj = bj 
        # Ensure position is a NumPy array for vector operations
        self.position = np.array(position) 
        # power_max and power_min are added externally after initialization
        # They define the operational range for the agent's power action.
        self.power_max = None 
        self.power_min = None

    def received_power(self, grj, distance):
        """
        Calculates the jamming power received by a radar from this jammer.

        Uses a simplified Friis transmission equation variant:
        P_received = (P_jammer * G_jammer * G_radar_towards_jammer) / (distance^2 * L_jammer * L_atm * B_jammer)
        (Note: Original paper formula might include wavelength, 4pi factors, etc. 
         This is a simplified version based on common jamming models.)

        Args:
            grj (float): Radar's receiving antenna gain towards this jammer (linear units).
            distance (float): Distance between the jammer and the radar (meters).

        Returns:
            float: The received jamming power in Watts, or 0.0 if calculation is invalid.
        """
        # Ensure current power is within the defined operational range (non-negative)
        current_power = max(0.0, self.power) 
        # Clamp power if min/max are defined (although current logic doesn't enforce this strictly here)
        # if self.power_min is not None:
        #     current_power = max(self.power_min, current_power)
        # if self.power_max is not None:
        #     current_power = min(self.power_max, current_power)
            
        # Prevent division by zero or issues with extremely small distances
        distance_sq = max(1e-9, distance**2) # Use a small epsilon for numerical stability
        
        # Calculate the denominator including losses and bandwidth
        # Ensure bandwidth is positive to avoid division by zero or negative results
        effective_bj = max(1e-9, self.bj) 
        denominator = distance_sq * self.loss * self.latm * effective_bj
        
        # Handle potential division by zero if any linear factor becomes zero unexpectedly
        # or if the calculated denominator is extremely close to zero.
        if denominator <= 1e-18: 
             print(f"Warning: Near-zero denominator in received_power calculation ({denominator}). Returning 0.") # Optional warning
             return 0.0 
             
        # Calculate received power using the simplified formula
        received_jamming_power = (current_power * self.gj * grj) / denominator
        
        # Ensure the result is physically plausible (non-negative)
        return max(0.0, received_jamming_power)
