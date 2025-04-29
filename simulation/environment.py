import numpy as np
import yaml # For loading config
import os   # For path joining

from core.radar import Radar # Assuming Radar class is in core
from core.jammer import Jammer # Assuming Jammer class is in core
from utils.math_utils import calculate_distance # Assuming distance util exists
from utils.state_utils import one_hot_encode # Assuming one-hot util exists

# Default path for simulation config
DEFAULT_SIM_CONFIG_PATH = "config/simulation_config.yaml"

# TODO: Load these from a config file (e.g., simulation_config.yaml)
# DEFAULT_RADAR_PARAMS (removed, should come fully from config)
# DEFAULT_JAMMER_PARAMS (removed, should come fully from config)

# Placeholder: Maximum number of radar types for one-hot encoding
# This should ideally be determined from the config or a global constant.
# MAX_RADAR_TYPES = 4 # REMOVED - Will load from config

# Reward component weights (placeholders - will need refinement based on doc)
# REWARD_WEIGHTS = { # REMOVED - Will load specific reward params from config
#     'detection_penalty': -1.0, # Placeholder
#     'power_cost': -0.01        # Placeholder
# }
# Fixed target signal power (placeholder - refine later)
FIXED_TARGET_SIGNAL_POWER = 1e-12

class ElectromagneticEnvironment:
    """
    Simulates the multi-jammer, multi-radar electromagnetic confrontation scenario.
    Handles state representation, action application, state transitions, 
    and reward calculation.
    """
    def __init__(self, config, sim_config_path=DEFAULT_SIM_CONFIG_PATH):
        """
        Initializes the environment.

        Args:
            config: A dictionary or SimpleNamespace containing RL agent parameters 
                    (e.g., from default.yaml - num_jammers, num_radars, episode_limit).
            sim_config_path (str): Path to the YAML file containing simulation entity parameters.
        """
        # Load simulation parameters (radars, jammers) from YAML
        try:
            with open(sim_config_path, 'r') as f:
                self.sim_config = yaml.safe_load(f)
            if not self.sim_config or 'radars' not in self.sim_config or 'jammers' not in self.sim_config:
                 raise ValueError(f"Simulation config file {sim_config_path} is missing required 'radars' or 'jammers' keys.")
        except FileNotFoundError:
             print(f"Error: Simulation configuration file not found at {sim_config_path}")
             raise
        except yaml.YAMLError as e:
             print(f"Error parsing simulation configuration file {sim_config_path}: {e}")
             raise
             
        # Ensure 'protected_target' key exists in config
        if 'protected_target' not in self.sim_config:
            raise ValueError(f"Simulation config file {sim_config_path} is missing required 'protected_target' key.")
        self.protected_target_config = self.sim_config['protected_target']
        if 'position' not in self.protected_target_config or 'rcs' not in self.protected_target_config:
            raise ValueError("Protected target config must contain 'position' and 'rcs'.")
        # Ensure target position is numpy array
        self.protected_target_config['position'] = np.array(self.protected_target_config['position'])
        
        # --- Load Environment Parameters (NEW) ---
        env_params = self.sim_config.get('environment_params', {})
        self.max_radar_types = env_params.get('max_radar_types', 4) # Default to 4 if missing
        reward_params = env_params.get('rewards', {})
        self.rd_min_penalty = reward_params.get('rd_min', -1.2)
        self.rd_max_penalty = reward_params.get('rd_max', -0.8)
        self.rp_min_penalty = reward_params.get('rp_min', -0.1)
        self.rp_max_penalty = reward_params.get('rp_max', -0.01)
        
        # Validate reward params make sense (min penalty should be more negative than max penalty)
        if self.rd_min_penalty >= self.rd_max_penalty:
             print(f"Warning: rd_min ({self.rd_min_penalty}) should be less than rd_max ({self.rd_max_penalty}) in config.")
        if self.rp_min_penalty >= self.rp_max_penalty:
             print(f"Warning: rp_min ({self.rp_min_penalty}) should be less than rp_max ({self.rp_max_penalty}) in config.")
             
        # Get counts and limits from the main RL config 
        self.num_jammers = getattr(config, 'num_jammers', len(self.sim_config.get('jammers', [])))
        self.num_radars = getattr(config, 'num_radars', len(self.sim_config.get('radars', [])))
        self.episode_limit = getattr(config, 'episode_limit', 100)

        # --- Action Space Definition (Aligned with Documentation) ---
        # Discrete T_i: 0=idle, 1..K (K=2*num_radars) represents target/type.
        # - mod(T_i, 2) == 0 -> Deception Jamming (for T_i > 0)
        # - mod(T_i, 2) == 1 -> Suppression Jamming
        # - Target Radar Index = floor((T_i + 1) / 2) - 1 (0-based)
        self.action_dim_discrete = 2 * self.num_radars + 1 
        self.action_dim_continuous = 1 # Normalized power level P_i [0, 1]
        
        # --- State/Observation Space Dimension Calculation (Based on Documentation) ---
        # Radar features: Pt, theta_m, Ts, Type_r (one-hot), theta_a, pos_r (x, y)
        # Jammer features: pos_j (x, y)
        radar_feature_dim = 1 + 1 + 1 + self.max_radar_types + 1 + 2 # Use loaded max_radar_types
        jammer_feature_dim = 2 # x, y position per jammer
        self.state_dim = self.num_radars * radar_feature_dim + self.num_jammers * jammer_feature_dim
        self.obs_dim = self.state_dim # Assume full observability for now (obs = state)
        self.agent_obs_dim = self.obs_dim # Each agent gets the full state 

        # Initialize internal state variables FIRST
        self._step_count = 0
        # Last actions format needs reconsideration if action space changes
        self._last_actions = np.zeros((self.num_jammers, 2)) 
        # _current_radar_pd is no longer part of the primary state vector
        # self._current_radar_pd = np.ones(self.num_radars)
        
        # Initialize radar/jammer entities using loaded sim_config
        # This now needs to handle the new radar parameters
        self._initialize_entities() 
        
        # Print initialization summary ONCE
        print(f"Environment Initialized: {self.num_jammers} Jammers, {self.num_radars} Radars (from {sim_config_path})")
        print(f"State Dimension: {self.state_dim} (Radar features: {radar_feature_dim}, Jammer features: {jammer_feature_dim}, Max Radar Types: {self.max_radar_types})") # Added max types info
        print(f"Observation Dimension (per agent): {self.agent_obs_dim}")
        print(f"Action Dimension (Discrete): {self.action_dim_discrete}")
        print(f"Protected Target: Pos={self.protected_target_config['position']}, RCS={self.protected_target_config['rcs']}")
        print(f"Episode Limit: {self.episode_limit}")
        print(f"Reward Params: rd=[{self.rd_min_penalty}, {self.rd_max_penalty}], rp=[{self.rp_min_penalty}, {self.rp_max_penalty}]") # Added reward params info

    def _initialize_entities(self):
        """Initializes radar and jammer properties from loaded sim_config."""
        self.radars = []
        self.jammers = []

        # Load radars
        radar_configs = self.sim_config.get('radars', [])[:self.num_radars]
        if len(radar_configs) < self.num_radars:
             print(f"Warning: Requested {self.num_radars} radars, but only {len(radar_configs)} found in config.")

        for i, params in enumerate(radar_configs):
            try:
                # Check mandatory parameters (including Ga and D)
                required_radar_params = [
                    'pt', 'gt', 'gr', 'wavelength', 'rcs', 'loss', 'latm', 'pn', 
                    'type_id', 'position', 'theta_m', 'theta_a', 't_s',
                    'pulse_compression_gain', 'anti_jamming_factor' # Added Ga, D
                ]
                for p_name in required_radar_params:
                     if p_name not in params:
                          raise KeyError(f"Radar config {i} missing required parameter: {p_name}")
                
                threat_level = params.get('threat_level', 1.0)
                if not (0 <= params.get('type_id', -1) < self.max_radar_types):
                    raise ValueError(f"Radar config {i} invalid type_id")

                # Create radar object (pass Ga and D)
                radar_obj = Radar(
                    pt=params['pt'], gt=params['gt'], gr=params['gr'], 
                    wavelength=params['wavelength'], rcs=params['rcs'], 
                    loss=params['loss'], latm=params['latm'], pn=params['pn'], 
                    type_id=params['type_id'], position=params['position'], 
                    theta_m=params['theta_m'], theta_a=params['theta_a'], t_s=params['t_s'],
                    pulse_compression_gain=params['pulse_compression_gain'], # Pass Ga
                    anti_jamming_factor=params['anti_jamming_factor'],     # Pass D
                    threat_level=threat_level 
                )
                self.radars.append(radar_obj)
            except KeyError as e:
                 print(f"Error initializing radar {i}: Missing parameter {e} in simulation_config.yaml")
                 raise
            except ValueError as e:
                 print(f"Error initializing radar {i}: Invalid value - {e}")
                 raise
            except Exception as e:
                 print(f"Error initializing radar {i}: {e}")
                 raise

        # Load jammers
        jammer_configs = self.sim_config.get('jammers', [])[:self.num_jammers]
        if len(jammer_configs) < self.num_jammers:
             print(f"Warning: Requested {self.num_jammers} jammers, but only {len(jammer_configs)} found in config.")

        for i, params in enumerate(jammer_configs):
            try:
                 # Check mandatory parameters
                 required_jammer_params = ['gj', 'loss', 'latm', 'bj', 'position']
                 for p_name in required_jammer_params:
                     if p_name not in params:
                         raise KeyError(f"Jammer config {i} missing required parameter: {p_name}")
                         
                 power_max = params.get('power_max', 100.0)
                 power_min = params.get('power_min', 0.0)
                 init_params = params.copy()
                 init_params['power'] = 0.0
                 init_params.pop('power_max', None) 
                 init_params.pop('power_min', None)
                 jammer_obj = Jammer(**init_params)
                 jammer_obj.power_max = power_max
                 jammer_obj.power_min = power_min
                 self.jammers.append(jammer_obj)
            except KeyError as e:
                 print(f"Error initializing jammer {i}: Missing parameter {e} in simulation_config.yaml")
                 raise
            except Exception as e:
                 print(f"Error initializing jammer {i}: {e}")
                 raise
            
        # print(f"Initialized {len(self.radars)} Radars and {len(self.jammers)} Jammers from config.") # MOVED to __init__
        # print(f"Protected Target: Pos={self.protected_target_config['position']}, RCS={self.protected_target_config['rcs']}") # MOVED to __init__
        self._step_count = 0
        self._last_actions.fill(0)
        # We need to ensure initial state is consistent
        # self._current_radar_pd is removed

    def reset(self):
        """
        Resets the environment to an initial state.

        Returns:
            np.ndarray: The initial global state observation.
        """
        # Re-initialize entities and state based on config
        # This ensures radars reset their positions, theta_a etc. if needed
        # Or, we might need a more specific reset logic for dynamic parts like theta_a later.
        self._initialize_entities() 
        return self.get_state() # Return the full state as the initial observation

    def step(self, actions):
        """
        Executes one time step in the environment.
        Args:
            actions (list): List of actions [(discrete_action_T_i, continuous_power_P_i), ...]
                            T_i: Discrete action index (0 to 2*num_radars)
                            P_i: Continuous power level (0 to 1)
        Returns:
            tuple: (next_state, reward, terminated, info)
        """
        # --- (Input validation) ---
        if len(actions) != self.num_jammers:
             raise ValueError(f"Received {len(actions)} actions, but expected {self.num_jammers}")
             
        self._step_count += 1
        
        # --- Update Radar States --- 
        # TODO: Implement radar behavior (e.g., scanning)

        # --- Apply Jammer Actions and Calculate Effects --- 
        total_suppression_power_per_radar = np.zeros(self.num_radars) # Watts, for Prjs
        # Need separate tracking for deception for r_j calculation later
        deception_actions_info = [] # Store info needed for r_j calculation later
        jammer_power_cost_penalty_terms = np.zeros(self.num_jammers) # For r_p
        jammer_actions_details = [] # For general info/debugging
        
        current_actions_array = np.zeros_like(self._last_actions) # Store raw actions
        for i, jammer in enumerate(self.jammers):
            discrete_action_T_i, continuous_power_P_i = actions[i]
            discrete_action_T_i = int(discrete_action_T_i) # Ensure integer
            continuous_power_P_i = np.clip(continuous_power_P_i, 0.0, 1.0)
            
            current_actions_array[i, 0] = discrete_action_T_i
            current_actions_array[i, 1] = continuous_power_P_i
            
            # Decode discrete action T_i according to documentation
            target_radar_idx = -1 # Default: No target (idle)
            jamming_type = None   # None, 0 (Deception), 1 (Suppression)
            is_jamming = False
            
            if discrete_action_T_i > 0:
                 # Check if action index is valid
                 if 1 <= discrete_action_T_i <= 2 * self.num_radars:
                      target_radar_idx = (discrete_action_T_i + 1) // 2 - 1 # Calculate 0-based target index
                      jamming_type = discrete_action_T_i % 2 # 0 for Deception (even Ti>0), 1 for Suppression (odd Ti)
                      is_jamming = True
                 else:
                      print(f"Warning: Jammer {i} chose invalid discrete action T_i={discrete_action_T_i}")
            
            # Scale normalized power to actual power P_j
            actual_power = jammer.power_min + continuous_power_P_i * (jammer.power_max - jammer.power_min)
            jammer.power = actual_power # Update jammer's current power attribute
            
            # Calculate contribution to r_p penalty term (normalized power used)
            power_range = jammer.power_max - jammer.power_min
            normalized_power_term = (actual_power - jammer.power_min) / power_range if power_range > 1e-6 else 0.0
            jammer_power_cost_penalty_terms[i] = normalized_power_term

            # If jamming a valid target radar with positive power:
            if is_jamming and actual_power > 0 and 0 <= target_radar_idx < self.num_radars:
                target_radar = self.radars[target_radar_idx]
                distance = calculate_distance(jammer.position, target_radar.position)
                
                if distance > 1e-6: 
                    grj = target_radar.get_antenna_gain(jammer.position)
                    received_j_power = jammer.received_power(grj=grj, distance=distance)
                    
                    action_detail = {
                        'jammer_idx': i,
                        'target_idx': target_radar_idx,
                        'type': jamming_type, # 0=Deception, 1=Suppression
                        'power': actual_power,
                        'received_power': received_j_power
                    }
                    jammer_actions_details.append(action_detail)
                    
                    # Accumulate power based on type
                    if jamming_type == 1: # Suppression
                        total_suppression_power_per_radar[target_radar_idx] += received_j_power
                    elif jamming_type == 0: # Deception
                        # Store info needed for r_j calculation later (e.g., received power Prjd)
                        deception_actions_info.append(action_detail)
            
        self._last_actions = current_actions_array # Store the raw actions taken

        # --- Simulate Detections & Update Radar States --- 
        radar_detection_results = [] 
        current_radar_pd = np.zeros(self.num_radars)
        true_target_snr_no_jamming = np.zeros(self.num_radars) # Store for r_j calc
        true_target_snr_with_jamming = np.zeros(self.num_radars) # Store for r_j calc
        
        # Get protected target info
        target_pos = self.protected_target_config['position']
        target_rcs = self.protected_target_config['rcs']
        
        for r_idx, radar in enumerate(self.radars):
            noise_power = radar.pn_watts
            suppressive_interference = total_suppression_power_per_radar[r_idx] 
            anti_jamming_factor_D = radar.anti_jamming_factor
            pulse_gain_Ga = radar.pulse_compression_gain
            
            # Calculate true target echo power Ps at this radar
            echo_power_Ps = radar.calculate_echo_power(target_pos, target_rcs)
            
            # Calculate SNR WITHOUT jamming (for r_j suppression baseline)
            snr_no_jam = (pulse_gain_Ga * echo_power_Ps) / noise_power if noise_power > 1e-18 else 0.0
            true_target_snr_no_jamming[r_idx] = max(0.0, snr_no_jam)
            
            # Calculate SNR WITH suppressive jamming (SNR_a from docs)
            # SNR_a = (Ga * Ps) / (D * Prjs + Pn)
            denominator_snr_a = (anti_jamming_factor_D * suppressive_interference) + noise_power
            snr_with_jam = (pulse_gain_Ga * echo_power_Ps) / denominator_snr_a if denominator_snr_a > 1e-18 else 0.0
            true_target_snr_with_jamming[r_idx] = max(0.0, snr_with_jam)

            # Calculate detection probability based on SNR *with* jamming
            # This Pd is used for state transitions and potentially r_j suppression
            pd = radar.detection_probability(snr_with_jam) # Assumes snr_with_jam is linear ratio
            current_radar_pd[r_idx] = pd
            
            # Simulate actual detection outcome for state update
            detected_in_step = np.random.rand() <= pd
            detection_result = {
                 'detected': detected_in_step, 
                 'pd': pd, 
                 'snr': snr_with_jam,
                 'target_info': self.protected_target_config if detected_in_step else None # Store target info if detected
            }
            radar_detection_results.append(detection_result)
            radar.update_state(detection_result)

        # --- Calculate Reward (r = r_d + r_p + r_j) --- 
        # Calculate r_d (Tracking Penalty) - Using updated radar states
        r_d_total = 0.0
        # Documentation: "applied when a radar detects and locks onto a defense unit."
        # "absolute value is larger for radars with higher threat levels." (-1.2 to -0.8)
        # Mapping threat_level (e.g., 0.8-1.2) to penalty (-1.2 to -0.8)
        # Example linear mapping: penalty = -1.0 + (threat_level - 1.0) * 0.2 / 0.2 = -1.0 + (threat_level - 1.0) = threat_level - 2.0? No.
        # Let's try simple: penalty = - threat_level. If threat is 1.2, penalty is -1.2. If threat is 0.8, penalty is -0.8.
        for r_idx, radar in enumerate(self.radars):
            radar_state_info = radar.get_state_info()
            if radar_state_info['is_tracking']: # Check if radar is in TRACK state
                 penalty = -radar_state_info['threat'] # Use threat level directly as penalty magnitude
                 # Ensure penalty is within the specified range [-1.2, -0.8] (or whatever range the config implies)
                 # Clamping might be needed if threat_level is outside [0.8, 1.2]
                 clamped_penalty = np.clip(penalty, self.rd_min_penalty, self.rd_max_penalty) # Use loaded config values
                 r_d_total += clamped_penalty
                 
        # Calculate r_p (Resource Consumption Penalty)
        # r_p_min = -0.1 # OLD Hardcoded
        # r_p_max = -0.01 # OLD Hardcoded
        r_p_total = 0.0
        for norm_power_term in jammer_power_cost_penalty_terms:
             # r_p_jammer = -r_p_max - (r_p_max - r_p_min) * norm_power_term # OLD Hardcoded
             # Note: rp_min_penalty is when power is max, rp_max_penalty when power is min
             # Formula is: penalty = min_penalty_at_zero_power + (max_penalty_at_full_power - min_penalty_at_zero_power) * normalized_power
             # In our config: rp_max is penalty at P=0, rp_min is penalty at P=max
             r_p_jammer = self.rp_max_penalty + (self.rp_min_penalty - self.rp_max_penalty) * norm_power_term 
             r_p_total += r_p_jammer
             
        # Calculate r_j (Jamming Success Reward)
        r_j_total = 0.0
        
        # 3.4: r_j for Suppression Jamming Actions
        # Defined as reduction in detection probability: P_d(no_jam) - P_d(with_jam)
        pd_no_jam = np.array([radar.detection_probability(snr) for radar, snr in zip(self.radars, true_target_snr_no_jamming)])
        pd_with_jam = current_radar_pd # Already calculated based on snr_with_jam
        pd_reduction = pd_no_jam - pd_with_jam
        
        # Sum the reduction for radars targeted by suppression jamming in this step
        # Need to know which radars were targeted by suppression
        suppression_targets = set() 
        for action_info in jammer_actions_details:
            if action_info['type'] == 1: # Suppression type
                 suppression_targets.add(action_info['target_idx'])
                 
        for target_idx in suppression_targets:
            if 0 <= target_idx < self.num_radars:
                 r_j_total += max(0.0, pd_reduction[target_idx]) # Add positive reduction as reward
                 
        # 3.5: r_j for Deception Jamming Actions
        # Needs: Calculate false target SNR_f = D * Prjd / Pn
        #        Calculate Pd for false targets (p_rd,f)
        #        Identify set F (detected false targets based on Monte Carlo sim)
        #        Calculate reward contribution per radar: 1 - product(1 - p_rd,f) for f in F
        # Accumulate reward contributions for each radar targeted by deception.

        # Store detected false target probabilities per radar
        detected_false_pd_per_radar = {r_idx: [] for r_idx in range(self.num_radars)}

        for action_info in deception_actions_info:
            target_idx = action_info['target_idx']
            if not (0 <= target_idx < self.num_radars):
                continue # Skip if target index is invalid
                
            radar = self.radars[target_idx]
            received_deception_power_Prjd = action_info['received_power']
            noise_power_Pn = radar.pn_watts
            anti_jamming_factor_D = radar.anti_jamming_factor
            
            # Calculate False Target SNR (SNR_f = D * Prjd / Pn)
            snr_f = (anti_jamming_factor_D * received_deception_power_Prjd) / noise_power_Pn if noise_power_Pn > 1e-18 else 0.0
            snr_f = max(0.0, snr_f)
            
            # Calculate detection probability for this false signal
            pd_f = radar.detection_probability(snr_f)
            
            # Simulate if this false target is detected (passes detection threshold)
            # Documentation: "If any false target in set F ... passes detection" 
            # Let's use Monte Carlo for now, consistent with real target detection sim.
            is_false_target_detected = np.random.rand() <= pd_f
            
            if is_false_target_detected:
                 # Add the Pd of the detected false target to the list for this radar
                 detected_false_pd_per_radar[target_idx].append(pd_f)
                 
        # Calculate the reward contribution for each radar targeted by deception
        r_j_deception = 0.0
        for target_idx, pd_list in detected_false_pd_per_radar.items():
            if not pd_list: # No detected false targets for this radar
                 continue
                 
            # Calculate product term: product(1 - p_rd,f) for f in F (detected false targets)
            product_term = 1.0
            for pd_f in pd_list:
                 # Clamp pd_f slightly below 1 to avoid log(0) issues if 1-pd=0 later, though unlikely.
                 safe_pd_f = min(pd_f, 0.999999)
                 product_term *= (1.0 - safe_pd_f)
                 
            # Reward contribution for this radar = 1 - product_term
            radar_deception_reward = 1.0 - product_term
            r_j_deception += radar_deception_reward
            
        # Add deception reward to total r_j
        r_j_total += r_j_deception

        # Final Reward
        global_reward = r_d_total + r_p_total + r_j_total
        
        # --- Termination Condition --- 
        terminated = self._step_count >= self.episode_limit
        
        # --- Information Dictionary --- 
        radar_states = [r.get_state_info() for r in self.radars]
        info = {
            'radar_pds': current_radar_pd, 
            'radar_states': radar_states,
            'snr_no_jamming': true_target_snr_no_jamming, 
            'snr_with_jamming': true_target_snr_with_jamming, 
            'r_d': r_d_total,
            'r_p': r_p_total,
            'r_j': r_j_total, # Now includes suppression and deception
            'jammer_actions': jammer_actions_details
        } 

        # --- Return Values --- 
        next_obs_list = self.get_obs() # Get list of observations for next step
        return next_obs_list, global_reward, terminated, info

    def get_state(self):
        """
        Computes the global state vector according to the documentation.
        s = [ (P_t, theta_m, T_s, Type_r_onehot, theta_a, pos_r_x, pos_r_y)_i , (pos_j_x, pos_j_y)_j ]

        Returns:
            np.ndarray: The global state vector.
        """
        state_features = []

        # Radar features
        for radar in self.radars:
            # 1. P_t (Peak Transmit Power)
            state_features.append(radar.pt)
            # 2. theta_m (Main Beam Width)
            state_features.append(radar.theta_m)
            # 3. T_s (Search Scan Period)
            state_features.append(radar.t_s)
            # 4. Type_r (One-Hot Encoded)
            type_one_hot = one_hot_encode(radar.type_id, self.max_radar_types) # Use loaded value
            state_features.extend(type_one_hot)
            # 5. theta_a (Main Beam Direction)
            state_features.append(radar.theta_a)
            # 6. pos_r (Flattened Position)
            state_features.extend(radar.position.flatten())
            
        # Jammer features
        for jammer in self.jammers:
            # 7. pos_j (Flattened Position)
            state_features.extend(jammer.position.flatten())

        return np.array(state_features, dtype=np.float32)

    def get_obs(self):
        """
        Returns the observation for each agent.
        Currently assumes full observability (each agent gets the global state).

        Returns:
            list[np.ndarray]: A list containing the observation for each agent (global state).
        """
        global_state = self.get_state()
        # Return a list where each element is the global state
        return [global_state for _ in range(self.num_jammers)]

    def get_agent_obs(self, agent_id):
        """
        Returns the observation for a specific agent.
        Assumes full observability.

        Args:
            agent_id (int): The index of the jammer agent.

        Returns:
            np.ndarray: The observation for the specified agent (global state).
        """
        if not (0 <= agent_id < self.num_jammers):
             raise ValueError(f"Invalid agent_id {agent_id} for {self.num_jammers} jammers.")
        return self.get_state()
        
    def get_avail_actions(self):
        """
        Returns the available actions for each agent.
        Assumes all actions (0 to 2*num_radars) are always available.
        
        Returns:
            list[np.ndarray]: List of available actions (binary mask) for each agent.
        """
        avail_actions = []
        for _ in range(self.num_jammers):
            # Assumes all discrete actions (size = 2*num_radars+1) are available
            avail_actions.append(np.ones(self.action_dim_discrete, dtype=np.int32))
        return avail_actions

    def get_env_info(self):
        """
        Provides environment information needed by the RL runner/learner.
        Returns:
            dict: Dictionary containing environment details.
        """
        return {
            "state_shape": self.state_dim,
            "obs_shape": self.state_dim, # Shape of observation for ONE agent
            "n_actions": self.action_dim_discrete, 
            "n_agents": self.num_jammers,
            "episode_limit": self.episode_limit
        }

    def close(self):
        """
        Clean up any resources (if needed).
        """
        print("Closing Electromagnetic Environment.")
        # No specific resources to close currently
        pass

# Example Usage (if run directly)
if __name__ == '__main__':
    print("Testing ElectromagneticEnvironment...")
    
    # Create dummy RL config
    class DummyConfig:
        num_jammers = 2
        num_radars = 2
        episode_limit = 10
        
    config = DummyConfig()
    
    # Create dummy simulation config file
    dummy_sim_config = {
        'radars': [
            {'pt': 1e6, 'gt': 30, 'gr': 30, 'wavelength': 0.03, 'rcs': 1.0, 'loss': 10, 'latm': 2, 'pn': -90, 'type_id': 0, 'position': [10000, 0], 'theta_m': 2.0, 'theta_a': 0.0, 't_s': 5.0, 'pulse_compression_gain': 100.0, 'anti_jamming_factor': 10.0, 'threat_level': 0.8},
            {'pt': 1.2e6, 'gt': 32, 'gr': 32, 'wavelength': 0.03, 'rcs': 1.5, 'loss': 8, 'latm': 2, 'pn': -92, 'type_id': 1, 'position': [-10000, 5000], 'theta_m': 1.8, 'theta_a': 180.0, 't_s': 4.0, 'pulse_compression_gain': 120.0, 'anti_jamming_factor': 15.0, 'threat_level': 1.2}
        ],
        'jammers': [
            {'power_max': 100, 'power_min': 0, 'gj': 20, 'loss': 5, 'latm': 2, 'bj': 1e6, 'position': [0, 1000]},
            {'power_max': 120, 'power_min': 10, 'gj': 22, 'loss': 4, 'latm': 2, 'bj': 1.2e6, 'position': [0, -1000]}
        ],
        'protected_target': {
             'position': [0, 0], # Example target at origin
             'rcs': 2.0 # Example RCS
        },
        'environment_params': {
            'max_radar_types': 4,
            'rewards': {
                'rd_min': -1.2,
                'rd_max': -0.8,
                'rp_min': -0.1,
                'rp_max': -0.01
            }
        }
    }
    test_config_path = "config/test_simulation_config.yaml"
    # Create config directory if it doesn't exist
    os.makedirs(os.path.dirname(test_config_path), exist_ok=True)
    with open(test_config_path, 'w') as f:
        yaml.dump(dummy_sim_config, f)
        
    # Create dummy util functions if they dont exist
    if not os.path.exists("utils/math_utils.py"):
         os.makedirs("utils", exist_ok=True)
         with open("utils/math_utils.py", "w") as f:
              f.write("import numpy as np\n")
              f.write("def db_to_linear(db): return 10**(db / 10)\n")
              f.write("def calculate_distance(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))\n")
              
    if not os.path.exists("utils/state_utils.py"):
         os.makedirs("utils", exist_ok=True)
         with open("utils/state_utils.py", "w") as f:
              f.write("import numpy as np\n")
              f.write("def one_hot_encode(index, num_classes):\n")
              f.write("    if not (0 <= index < num_classes): raise ValueError(f'Index {index} out of bounds for {num_classes} classes')\n")
              f.write("    encoded = np.zeros(num_classes)\n")
              f.write("    encoded[index] = 1.0\n")
              f.write("    return encoded\n")
              
    # Need dummy core files too
    if not os.path.exists("core/radar.py") or not os.path.exists("core/jammer.py"):
        print("Error: Need core/radar.py and core/jammer.py for testing.")
        # Maybe create minimal versions if needed?
    else:
        env = ElectromagneticEnvironment(config, sim_config_path=test_config_path)
        initial_state = env.reset()
        print(f"Initial State (shape {initial_state.shape}):\n{initial_state}")
        env_info = env.get_env_info()
        print(f"\nEnv Info: {env_info}") # Check n_actions

        # Test multiple steps to see state transitions and r_d
        print("\n--- Running Test Steps ---")
        state = env.reset()
        total_reward = 0
        for step in range(5):
             # Example: J0 deceives R0 (T=2), J1 suppresses R1 (T=3)
             actions = [(2, 0.7), (3, 0.9)] 
             print(f"\nStep {step+1} Actions: {actions}")
             next_state, reward, terminated, info = env.step(actions)
             print(f"Step {step+1} Reward: {reward:.4f}")
             print(f"Step {step+1} Info: {info}") 
             state = next_state
             total_reward += reward
             if terminated:
                  print("Episode terminated early.")
                  break
                  
        print(f"\nTotal Reward over {step+1} steps: {total_reward:.4f}")

        env.close()
        print("\nTest finished.")
        # Clean up dummy config file
        # os.remove(test_config_path) 