"""
Main execution script for training the Multi-Agent Cooperative Jamming Decision (MA-CJD) agent
using the QMix algorithm in the Electromagnetic Environment simulator.

This script handles:
1. Loading configuration from YAML files (default.yaml, potentially others).
2. Setting up the computation device (CPU/GPU) and random seeds.
3. Initializing the simulation environment (ElectromagneticEnvironment).
4. Initializing core MARL components:
    - Multi-Agent Controller (BasicMAC)
    - QMix Learner (QMixLearner)
    - Replay Buffer (EpisodeReplayBuffer)
    - Episode Runner (EpisodeRunner)
5. Setting up TensorBoard logging for monitoring training progress.
6. Running the main training loop:
    - Collecting episode rollouts using the EpisodeRunner.
    - Storing episode data in the ReplayBuffer.
    - Sampling batches from the buffer and training the QMixLearner.
    - Periodically logging performance metrics (return, loss, epsilon, etc.) to console and TensorBoard.
    - Periodically saving model checkpoints.
7. Closing the environment and TensorBoard writer upon completion.
"""
import torch
import numpy as np
import yaml
import argparse
import os
import time
from types import SimpleNamespace
from datetime import datetime # For unique log dirs

# --- Import Core Components ---
# Environment simulation
from simulation.environment import ElectromagneticEnvironment 
# Agent controller (selects actions based on observations)
from core.mac import BasicMAC
# Learning algorithm (trains agents and mixer)
from core.qmix import QMixLearner
# Stores and samples episode data
from utils.replay_buffer import EpisodeReplayBuffer
# Handles interaction between agent, env, and buffer for one episode
from runners.episode_runner import EpisodeRunner

# --- Utilities ---
# For tracking recent stats efficiently
from collections import deque
# For writing logs viewable in TensorBoard
from torch.utils.tensorboard import SummaryWriter

def load_config(config_name="default", config_dir="config"):
    """
    Loads configuration parameters from a specified YAML file name in the config directory.

    Args:
        config_name (str): Base name of the YAML configuration file (without .yaml).
        config_dir (str): Directory where config files are stored.

    Returns:
        SimpleNamespace: A namespace object containing the loaded configuration parameters.
                         Allows accessing parameters using dot notation (e.g., config.lr).

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    path = os.path.join(config_dir, f"{config_name}.yaml")
    try:
        with open(path, 'r') as f:
            # Load the YAML file into a dictionary
            config_dict = yaml.safe_load(f)
        # Convert the dictionary to a SimpleNamespace for convenient access
        print(f"Configuration loaded successfully from {path}")
        return SimpleNamespace(**config_dict)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file {path}: {e}")
        raise

def run(args):
    """ 
    Main training function, takes parsed args.
    Initializes all components and runs the training loop.

    Args:
        args (SimpleNamespace): Configuration object containing all necessary parameters from parsed args and config file.
                                Note: This object is named 'config' in the calling scope.
    """
    # --- Setup ---
    # Determine computation device based on request stored in the config object
    requested_device = args.device_request.lower()
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if requested_device == "cuda":
             print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
        
    # Store the actual device string used back into the config object (args)
    args.device = str(device) 
    args.use_cuda = (device.type == 'cuda') # Update use_cuda flag based on actual device
    print(f"Using device: {args.device}") # Print the actual device used

    # Set random seeds for reproducibility across NumPy and PyTorch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        # Double-check CUDA availability
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            args.use_cuda = False # Update args flag
            args.device = "cpu"   # Update args device
            device = torch.device("cpu") # Update local device variable
        else:
            # Set CUDA seed if using GPU
            torch.cuda.manual_seed(args.seed)
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA current device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # --- TensorBoard Logging Setup ---
    # Use the provided test_name or default
    test_name = args.test_name if hasattr(args, 'test_name') else "ma_cjd_test"
    # Create a unique run name based on timestamp
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Define the directory to save logs, nested under the test name
    log_dir = os.path.join("logs", test_name, run_name)
    # Initialize the SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # --- Initialize Environment --- 
    # Construct the path for the simulation config file using the env_config arg
    sim_config_path = os.path.join("config", f"{args.env_config}.yaml")
    # Create the environment instance, passing the main config (args) and the sim config path
    env = ElectromagneticEnvironment(config=args, sim_config_path=sim_config_path) 
    # Get essential information from the environment (shapes, counts, limits)
    env_info = env.get_env_info()
    # Store environment info back into the main config object (args) for easy access
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    # Use observation shape if provided, otherwise default to state shape
    args.obs_shape = env_info.get("obs_shape", args.state_shape) 
    args.episode_limit = env_info["episode_limit"]
    args.env_info = env_info # Store the whole env_info dict as well

    # --- Initialize MARL Components --- 
    # Multi-Agent Controller (houses the agent network(s))
    mac = BasicMAC(input_shape=args.obs_shape, args=args)
    # Replay buffer (stores episodes)
    buffer = EpisodeReplayBuffer(args=args)
    # Learner (houses networks, optimizer, performs training steps)
    learner = QMixLearner(mac, args=args) 
    # Episode Runner (handles env stepping, action selection, data collection)
    runner = EpisodeRunner(env=env, mac=mac, buffer=buffer, args=args)

    # Move MAC networks explicitly to the determined device if using CUDA
    # Note: Learner handles moving its internal networks (eval/target agent, mixer)
    if args.use_cuda:
        mac.cuda() 

    # --- Training Loop --- 
    start_time = time.time()
    last_log_time = start_time # Used for time-based logging interval
    episode = 0 # Episode counter
    total_steps = 0 # Total environment steps taken across all episodes
    
    # Setup for storing recent statistics for logging averages
    log_stats = {
        "episode_return": deque(maxlen=args.log_interval),
        "episode_length": deque(maxlen=args.log_interval),
        "avg_step_reward": deque(maxlen=args.log_interval),
        "loss": deque(maxlen=args.log_interval * (args.episode_limit // args.train_interval)),
        # Add deques for new stats from learner
        "grad_norm": deque(maxlen=args.log_interval * (args.episode_limit // args.train_interval)),
        "eval_qtot_avg": deque(maxlen=args.log_interval * (args.episode_limit // args.train_interval)),
        "target_qtot_avg": deque(maxlen=args.log_interval * (args.episode_limit // args.train_interval)),
        # Add deques for reward components from runner info
        "reward_r_d": deque(maxlen=args.log_interval),
        "reward_r_p": deque(maxlen=args.log_interval),
        "reward_r_j": deque(maxlen=args.log_interval),
        # Add deques for action stats
        "avg_power": deque(maxlen=args.log_interval),
        "action_dist": deque(maxlen=args.log_interval) # Store distributions as numpy arrays
    }

    print("Starting training...")
    # Main loop continues until the desired total number of environment steps is reached
    while total_steps < args.total_env_steps:
        # --- Run One Episode ---
        run_info = runner.run(test_mode=False) 
        episode += 1
        current_episode_steps = run_info["episode_length"]
        total_steps += current_episode_steps
        
        # --- Store Episode Stats ---
        log_stats["episode_return"].append(run_info["episode_return"])
        log_stats["episode_length"].append(current_episode_steps)
        log_stats["avg_step_reward"].append(run_info.get("avg_step_reward", 0))
        # Log reward components if available in run_info (EpisodeRunner needs to collect them)
        log_stats["reward_r_d"].append(run_info.get("avg_r_d", 0))
        log_stats["reward_r_p"].append(run_info.get("avg_r_p", 0))
        log_stats["reward_r_j"].append(run_info.get("avg_r_j", 0))
        # Append action stats
        log_stats["avg_power"].append(run_info.get("avg_power_overall", 0))
        if "action_distribution" in run_info:
            log_stats["action_dist"].append(run_info["action_distribution"])

        # --- Train Learner ---
        if buffer.current_size >= args.batch_size and total_steps > args.start_training_steps:
            num_train_steps = current_episode_steps // args.train_interval
            episode_loss_sum = 0
            train_count = 0
            for _ in range(num_train_steps):
                batch = buffer.sample(args.batch_size)
                if batch is not None: 
                    train_stats = learner.train(batch, {'total_steps': total_steps})
                    # Log all stats returned by the learner
                    log_stats["loss"].append(train_stats['loss'])
                    log_stats["grad_norm"].append(train_stats['grad_norm'])
                    log_stats["eval_qtot_avg"].append(train_stats['eval_qtot_avg'])
                    log_stats["target_qtot_avg"].append(train_stats['target_qtot_avg'])
                    episode_loss_sum += train_stats['loss']
                    train_count += 1
            if train_count > 0:
                 writer.add_scalar('Loss/train_episode_avg', episode_loss_sum / train_count, total_steps)

        # --- Logging ---
        current_time = time.time()
        if (current_time - last_log_time) >= args.log_interval_seconds:
             avg_return = np.mean(log_stats["episode_return"]) if log_stats["episode_return"] else 0
             avg_length = np.mean(log_stats["episode_length"]) if log_stats["episode_length"] else 0
             avg_step_reward_log = np.mean(log_stats["avg_step_reward"]) if log_stats["avg_step_reward"] else 0
             avg_loss = np.mean(log_stats["loss"]) if log_stats["loss"] else 0
             elapsed_time = current_time - start_time
             # Calculate averages for new stats
             avg_grad_norm = np.mean(log_stats["grad_norm"]) if log_stats["grad_norm"] else 0
             avg_eval_qtot = np.mean(log_stats["eval_qtot_avg"]) if log_stats["eval_qtot_avg"] else 0
             avg_target_qtot = np.mean(log_stats["target_qtot_avg"]) if log_stats["target_qtot_avg"] else 0
             avg_r_d = np.mean(log_stats["reward_r_d"]) if log_stats["reward_r_d"] else 0
             avg_r_p = np.mean(log_stats["reward_r_p"]) if log_stats["reward_r_p"] else 0
             avg_r_j = np.mean(log_stats["reward_r_j"]) if log_stats["reward_r_j"] else 0
             # Calculate average action stats
             avg_power_log = np.mean(log_stats["avg_power"]) if log_stats["avg_power"] else 0
             # Average the distribution arrays
             avg_action_dist_log = np.mean(np.array(log_stats["action_dist"]), axis=0) if log_stats["action_dist"] else np.zeros(args.n_actions)
             # Format action distribution for printing
             action_dist_str = " / ".join([f"{p:.2f}" for p in avg_action_dist_log])

             print(f"Steps: {total_steps}/{args.total_env_steps} | Episodes: {episode} | Time: {elapsed_time:.2f}s")
             print(f"  Avg Return (last {len(log_stats['episode_return'])} eps): {avg_return:.2f} | Avg Length: {avg_length:.1f} | Avg Loss: {avg_loss:.4f}")
             print(f"  Avg Step Reward (last {len(log_stats['avg_step_reward'])} eps): {avg_step_reward_log:.4f}")
             print(f"  Avg Rewards (r_d/r_p/r_j): {avg_r_d:.4f} / {avg_r_p:.4f} / {avg_r_j:.4f}") # Print reward components
             print(f"  Avg QTot (Eval/Target): {avg_eval_qtot:.4f} / {avg_target_qtot:.4f} | Avg Grad Norm: {avg_grad_norm:.4f}") # Print new stats
             print(f"  Avg Power: {avg_power_log:.3f} | Action Dist: [{action_dist_str}] (0=Idle, 1=S0, 2=D0, ...)") # Print action stats
             print(f"  Buffer Size: {len(buffer)}")
             print(f"  Epsilon: {mac.action_selector.epsilon:.3f}")

             writer.add_scalar('Perf/Avg_Return', avg_return, total_steps)
             writer.add_scalar('Perf/Avg_Length', avg_length, total_steps)
             writer.add_scalar('Perf/Avg_Step_Reward', avg_step_reward_log, total_steps)
             writer.add_scalar('Loss/train_avg', avg_loss, total_steps)
             writer.add_scalar('Params/Epsilon', mac.action_selector.epsilon, total_steps)
             writer.add_scalar('Params/Buffer_Size', len(buffer), total_steps)
             # Add new stats to TensorBoard
             writer.add_scalar('Stats/grad_norm', avg_grad_norm, total_steps)
             writer.add_scalar('QValues/eval_qtot_avg', avg_eval_qtot, total_steps)
             writer.add_scalar('QValues/target_qtot_avg', avg_target_qtot, total_steps)
             writer.add_scalar('Rewards/r_d_avg', avg_r_d, total_steps)
             writer.add_scalar('Rewards/r_p_avg', avg_r_p, total_steps)
             writer.add_scalar('Rewards/r_j_avg', avg_r_j, total_steps)
             # Add action stats to TensorBoard
             writer.add_scalar('Perf/Avg_Power', avg_power_log, total_steps)
             # Log action distribution (can be complex to visualize, maybe log counts or key actions?)
             for act_idx, act_prob in enumerate(avg_action_dist_log):
                 writer.add_scalar(f'ActionDist/Action_{act_idx}', act_prob, total_steps)

             last_log_time = current_time

        # --- Save Model ---
        if args.save_model and (episode % args.save_interval == 0 or total_steps >= args.total_env_steps):
             if total_steps > args.start_training_steps:
                 # Construct save directory path, including test_name
                 save_dir = os.path.join(args.save_model_dir, test_name, f"step_{total_steps}")
                 os.makedirs(save_dir, exist_ok=True)
                 print(f"Saving model to {save_dir}")
                 learner.save_models(save_dir) 

    # --- Cleanup ---
    # Ensure the environment is closed properly (if it has a close method)
    if hasattr(env, 'close') and callable(env.close):
        env.close()
    # Close the TensorBoard writer
    writer.close()
    print("Training finished.")

# --- Script Entry Point ---
if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='MA-CJD QMix Training')
    # Argument for the main RL configuration file name
    parser.add_argument('--config', type=str, default="default", 
                        help='Name of the primary configuration file (e.g., default, qmix_setting1)')
    # Argument for the environment/scenario configuration file name
    parser.add_argument('--env-config', type=str, default="simulation_config", 
                        help='Name of the environment scenario configuration file (e.g., simulation_config, scenario1)')
    # Argument to specify computation device
    parser.add_argument('--device', type=str, default="cuda", 
                        help='Computation device to use ("cuda" or "cpu")')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # --- Load Main Configuration --- 
    # Load parameters from the specified primary config file
    config = load_config(config_name=args.config)
    # Add the command-line arguments (like env_config name) to the config object
    # This makes them easily accessible alongside other parameters
    config.config = args.config # Store which main config was used
    config.env_config = args.env_config # Store which env config was specified
    # Store the requested device separately
    config.device_request = args.device 
    
    # --- Start Training --- 
    # Pass the combined config object to the main run function
    run(config) 