import numpy as np
from utils.math_utils import db_to_linear

# Define Radar States as constants for clarity
RADAR_STATE_SEARCH = "SEARCH"
RADAR_STATE_CONFIRM = "CONFIRM" # If needed based on detailed model
RADAR_STATE_TRACK = "TRACK"

class Radar:
    def __init__(self, pt, gt, gr, wavelength, rcs, loss, latm, pn, type_id, position, theta_m, theta_a, t_s, pulse_compression_gain, anti_jamming_factor, threat_level=1.0, initial_state=RADAR_STATE_SEARCH):
        self.pt = float(pt)
        self.gt = db_to_linear(gt)
        self.gr = db_to_linear(gr)
        self.lambda_ = wavelength
        self.rcs = rcs
        self.loss = db_to_linear(loss)
        self.latm = db_to_linear(latm)
        self.pn = db_to_linear(pn)
        self.pn_watts = 10**((pn - 30) / 10)
        self.type_id = type_id
        self.position = np.array(position)
        self.theta_m = theta_m
        self.theta_a = theta_a
        self.t_s = t_s
        
        # --- Added Parameters for r_j Calculation ---
        self.pulse_compression_gain = pulse_compression_gain # Ga (linear scale assumed)
        self.anti_jamming_factor = anti_jamming_factor     # D (linear scale assumed)
        
        # --- State & Threat --- 
        self.state = initial_state # Current operational state (e.g., SEARCH, TRACK)
        self.threat_level = threat_level # Threat level associated with this radar (e.g., 0.8 to 1.2)
        self.target_locked = None # Store info about the locked target if state is TRACK

    def calculate_echo_power(self, target_position, target_rcs):
        """
        Calculates the echo power received from a specific target.
        Uses the radar equation with target-specific distance and RCS.
        
        Args:
            target_position (np.ndarray): Position [x, y] of the target.
            target_rcs (float): Radar Cross Section (m^2) of the target.
            
        Returns:
            float: Echo power Ps in Watts.
        """
        distance = np.linalg.norm(self.position - target_position)
        safe_distance = max(distance, 1e-6) # Avoid division by zero
        # Check if target is within main beam? For now, assume yes or use peak gains.
        # TODO: Refine gain calculation based on angle difference between theta_a and target direction.
        tx_gain = self.gt
        rx_gain = self.gr
        
        numerator = self.pt * tx_gain * rx_gain * (self.lambda_**2) * target_rcs
        denominator = ((4 * np.pi)**3) * (safe_distance**4) * self.loss * self.latm
        
        if denominator <= 1e-18:
             return 0.0 # Avoid division by zero
             
        return numerator / denominator

    def echo_power(self, distance):
        safe_distance = max(distance, 1e-6)
        return (self.pt * self.gt * self.gr * self.lambda_**2 * self.rcs) / \
               ((4 * np.pi)**3 * safe_distance**4 * self.loss * self.latm)

    def detection_probability(self, snr, prfa=1e-6, m=10):
        snr_linear = max(snr, 0.0)
        safe_prfa = max(prfa, 1e-18)
        A = np.log(0.62 / safe_prfa)
        safe_m = max(m, 1)
        log10_m = np.log10(safe_m) if safe_m > 0 else 0
        Z = snr_linear + (5 * log10_m) / (6.2 + 4.54 / np.sqrt(safe_m) + 0.44)
        denominator_B = 1.7 + 0.12 * A
        if abs(denominator_B) < 1e-9:
            return 0.0
        B = (10 * Z - A) / denominator_B
        if B > 700:
            return 1.0
        if B < -700:
            return 0.0
        return 1 / (1 + np.exp(-B))

    def get_antenna_gain(self, target_position):
        return self.gr

    def update_beam_direction(self, new_theta_a):
        self.theta_a = new_theta_a

    def update_state(self, detection_result):
        """
        Updates the radar's state based on the latest detection result.
        This is a simplified state machine for now.

        Args:
            detection_result (dict): Information about the detection attempt, 
                                     e.g., {'detected': bool, 'target_info': ..., 'snr': ...}
                                     (Structure needs to be defined based on simulation needs)
        """
        # Simple transitions: SEARCH -> TRACK on strong detection
        # More complex logic (CONFIRM state, track maintenance) could be added.
        if self.state == RADAR_STATE_SEARCH:
            if detection_result.get('detected', False):
                 # Add a condition, e.g., minimum SNR or confirmation steps? 
                 # For now, assume any detection leads to track attempt.
                 self.state = RADAR_STATE_TRACK
                 self.target_locked = detection_result.get('target_info', 'Unknown') # Store what was detected
            # Else: Remain in SEARCH
            
        elif self.state == RADAR_STATE_TRACK:
            # Condition to lose track? e.g., consecutive missed detections or low SNR
            if not detection_result.get('detected', False):
                 self.state = RADAR_STATE_SEARCH
                 self.target_locked = None
            # Else: Remain in TRACK, potentially update target info
            else:
                 self.target_locked = detection_result.get('target_info', self.target_locked) # Update if info changes
                 
        # Add logic for RADAR_STATE_CONFIRM if introduced later.

    def get_state_info(self):
        """
        Returns current state information.
        """
        return {
            'state': self.state,
            'threat': self.threat_level,
            'is_tracking': self.state == RADAR_STATE_TRACK,
            'locked_target': self.target_locked
        }
