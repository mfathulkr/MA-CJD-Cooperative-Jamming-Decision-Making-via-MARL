"""
Standalone script to run a single simulation step using the ElectromagneticEnvironment.

This script is intended for basic testing or demonstration of the environment 
independent of the main RL training loop (main.py). It loads simulation 
configurations, initializes the environment, executes one step with default 
or dummy actions (as no agent actions are provided here), and prints the 
resulting observations and simulation outcomes.
"""
import yaml
from simulation.environment import ElectromagneticEnvironment
from pprint import pprint
import numpy as np

def to_float_dict(d: dict):
    """Converts numeric-like values in a dictionary to float, skipping 'position'.
    
    Used during config loading to ensure numerical parameters are floats.
    """
    return {k: float(v) if isinstance(v, (int, float, str)) and k != "position" else v for k, v in d.items()}

def load_simulation_config(path="config/simulation_config.yaml"):
    """Loads radar and jammer configurations from a specified YAML file."""
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    # Ensure radar/jammer configurations are extracted and values converted
    radar_configs = [to_float_dict(radar) for radar in config.get("radars", [])]
    jammer_configs = [to_float_dict(jammer) for jammer in config.get("jammers", [])]

    # TODO: This function currently doesn't return the loaded sim_config itself, 
    #       only the processed radar/jammer lists. The environment requires more 
    #       than just these lists (e.g., protected_target, env_params).
    #       Consider returning the full config or modifying how the environment
    #       is initialized in main() if this script is to be fully functional.
    # return radar_configs, jammer_configs
    # Returning the full config for now, assuming Environment expects it or 
    # can handle it via its own config loading based on path.
    # Let's revert to the original return signature as Environment handles loading
    return radar_configs, jammer_configs # Original return

# convert numpy arrays to native Python types for better readability
def simplify_numpy(obj):
    """Recursively converts NumPy types within nested structures to native Python types."""
    if isinstance(obj, dict):
        return {k: simplify_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [simplify_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(simplify_numpy(v) for v in obj)
    elif isinstance(obj, np.generic):  # Catches np.float64, np.int64, etc.
        return obj.item()
    else:
        return obj

def main():
    """Main function to run a single environment step."""
    # Load simulation entity parameters (radars, jammers)
    # Note: The Environment class itself loads the full simulation config, 
    # including protected_target and environment_params. 
    # These loaded configs might not be directly used if the Environment
    # re-loads based on the path passed to it.
    # radar_configs, jammer_configs = load_simulation_config() 
    
    # Create a dummy main config object (similar to what main.py does)
    # The Environment needs info like num_jammers, num_radars, episode_limit.
    class DummyRLConfig:
        # Infer counts from a default config path or set defaults
        # Let's load the default simulation config to get counts
        try:
            with open("config/simulation_config.yaml", 'r') as f:
                _sim_cfg = yaml.safe_load(f)
            num_jammers = len(_sim_cfg.get('jammers', []))
            num_radars = len(_sim_cfg.get('radars', []))
            # Get episode limit from env_params or set a default
            _env_params = _sim_cfg.get('environment_params', {})
            episode_limit = _env_params.get('episode_limit', 50) # Default limit 50
        except Exception:
            print("Warning: Could not load simulation_config.yaml to determine agent/radar counts. Using defaults (2, 2).")
            num_jammers = 2
            num_radars = 2
            episode_limit = 50
            
    rl_config = DummyRLConfig()
    sim_config_path = "config/simulation_config.yaml" # Path to simulation details

    # Initialize the environment using the correct class and paths
    # Pass both the dummy RL config and the path to the simulation config
    env = ElectromagneticEnvironment(config=rl_config, sim_config_path=sim_config_path)

    # Get initial state/observation (reset is implicitly called in __init__)
    initial_obs = env.get_obs() 
    print("\n--- Initial Observations ---")
    pprint(simplify_numpy(initial_obs)) # Print simplified initial obs

    # Define dummy actions for the step (e.g., all jammers idle)
    # Action format: list of tuples [(discrete_action, continuous_param), ...]
    # Action 0 = Idle, Power 0.0
    dummy_actions = [(0, 0.0)] * env.num_jammers 
    print(f"\n--- Running Step with Actions: {dummy_actions} ---")

    # Run a single simulation step with the dummy actions
    observations, reward, terminated, info = env.step(dummy_actions)
    
    # Simplify results for printing
    observations_simple = simplify_numpy(observations)
    info_simple = simplify_numpy(info)

    # Print results from the single step
    print("\n--- Observations After Step ---")
    pprint(observations_simple)
    print("\n--- Info After Step ---")
    pprint(info_simple)
    print(f"\n--- Reward: {reward:.4f} ---")
    print(f"--- Terminated: {terminated} ---")

    # Example: Print specific info like radar detection probabilities
    print("\n--- Radar Detection Probabilities (Pd) ---")
    if 'radar_pds' in info_simple:
        for idx, pd in enumerate(info_simple['radar_pds']):
            print(f"Radar {idx}: Pd = {pd:.4f}")

if __name__ == "__main__":
    # This block ensures that main() is called only when the script 
    # is executed directly (e.g., `python simulation/run_simulation.py`), 
    # not when it's imported as a module.
    main()
