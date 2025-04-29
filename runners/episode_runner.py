import numpy as np
import torch
from functools import partial

class EpisodeRunner:
    """
    Handles running episodes, interaction between MAC and Env, and storing data.
    """
    def __init__(self, env, mac, buffer, args):
        self.env = env
        self.mac = mac
        self.buffer = buffer
        self.args = args
        self.episode_limit = self.env.get_env_info()["episode_limit"]
        self.t = 0 # Global timestep counter across episodes

        self.t_env = 0 # Total env steps
        
        # Get n_agents from env_info for processing observations correctly
        env_info = self.env.get_env_info()
        self.n_agents = env_info["n_agents"]
        # Use the obs_shape from env_info for the EpisodeBatch scheme
        self.new_batch = partial(EpisodeBatch, self.args, self.episode_limit, self.n_agents, env_info["obs_shape"])
        
        self.device = torch.device(args.device if torch.cuda.is_available() and args.use_cuda else "cpu")

    def run(self, test_mode=False):
        """ Runs a single episode. """
        batch = self.new_batch() # Create a new object to store episode data
        self.mac.init_hidden(batch_size=1) # Initialize hidden states for the MAC (batch_size=1 for rollout)
        
        terminated = False
        episode_return = 0
        step_rewards = [] # List to store rewards from each step
        # --- Add storage for reward components --- 
        ep_rewards_r_d = []
        ep_rewards_r_p = []
        ep_rewards_r_j = []
        # --- Add storage for action details ---
        ep_discrete_actions = [] # Store discrete actions for analysis
        ep_continuous_params = [] # Store continuous params for analysis
        # --- End Add ---
        self.env.reset()
        
        state = self.env.get_state()
        obs_list = self.env.get_obs() # Returns a list of obs arrays, one per agent

        step = 0
        while not terminated:
            # Get actions from MAC
            # Prepare inputs for MAC
            # Convert list of obs arrays into a single numpy array (n_agents, obs_dim)
            obs_np = np.array(obs_list)
            # Create tensor, add batch dim -> (1, n_agents, obs_dim)
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(self.device) 
            
            avail_actions_list = self.env.get_avail_actions() # List of avail_actions arrays
            avail_actions_np = np.array(avail_actions_list) 
            # Create tensor, add batch dim -> (1, n_agents, n_actions)
            avail_actions_tensor = torch.tensor(avail_actions_np, dtype=torch.long).unsqueeze(0).to(self.device)

            # select_actions expects batch, uses t_env for epsilon
            discrete_actions_tensor, continuous_params_tensor = self.mac.select_actions(obs_tensor, avail_actions_tensor, self.t_env, test_mode=test_mode)
            
            # Get the hidden state *after* the action selection for this step
            # This hidden state corresponds to the state used to compute Q(s,a)
            current_hidden_state_tensor = self.mac.hidden_states.detach().cpu() # Shape (1*n_agents, rnn_hidden_dim)
            # Reshape to (n_agents, rnn_hidden_dim) for storage
            current_hidden_state_np = current_hidden_state_tensor.reshape(self.n_agents, -1).numpy()

            # Convert actions to numpy for env step
            discrete_actions = discrete_actions_tensor.detach().squeeze(0).cpu().numpy() # Shape (n_agents, 1)
            continuous_params = continuous_params_tensor.detach().squeeze(0).cpu().numpy() # Shape (n_agents, 1)
            # --- Store action details for this step ---
            ep_discrete_actions.append(discrete_actions)
            ep_continuous_params.append(continuous_params)
            # --- End Store ---
            
            # Combine actions into the format expected by env.step()
            # List of (discrete_action, continuous_param) tuples
            actions_for_env = [(d[0], c[0]) for d, c in zip(discrete_actions, continuous_params)]

            # Environment step
            next_obs_list, reward, terminated, info = self.env.step(actions_for_env)
            
            step_rewards.append(reward)
            # --- Store reward components from info --- 
            ep_rewards_r_d.append(info.get('r_d', 0))
            ep_rewards_r_p.append(info.get('r_p', 0))
            ep_rewards_r_j.append(info.get('r_j', 0))
            # --- End Store --- 

            # We need current obs and state for the transition data
            # Convert current obs_list to array for storage
            current_obs_arr = np.array(obs_list) # Shape: (n_agents, obs_dim)
            current_state_arr = np.array(state) # Assuming state is already numpy array (state_dim,)
            # Convert current avail_actions list to array
            current_avail_actions_arr = np.array(avail_actions_list) # Shape: (n_agents, n_actions)
            
            episode_return += reward

            # Store transition data
            transition_data = {
                "state": current_state_arr,
                "obs": current_obs_arr,       
                "actions_discrete": discrete_actions, 
                "actions_continuous": continuous_params, 
                "avail_actions": current_avail_actions_arr,
                "reward": np.array([reward]), 
                "terminated": np.array([terminated]),
                "hidden_state": current_hidden_state_np
            }
            batch.push(transition_data)

            # Update current state/obs for the NEXT loop iteration
            state = self.env.get_state() 
            obs_list = next_obs_list # Update obs_list for the next iteration
            step += 1
            self.t_env += 1

            if terminated or step >= self.episode_limit:
                 last_state = self.env.get_state()
                 last_obs_list = next_obs_list
                 last_avail_actions_list = self.env.get_avail_actions()
                 last_data = {
                     "state": np.array(last_state),
                     "obs": np.array(last_obs_list),          
                     "avail_actions": np.array(last_avail_actions_list),
                 }
                 batch.update_last(last_data)
                 break
                 
        if not test_mode:
             self.buffer.store_episode(batch.get_batch_data())

        # Calculate average step reward for the episode
        avg_step_reward = np.mean(step_rewards) if step_rewards else 0
        # --- Calculate average reward components --- 
        avg_r_d = np.mean(ep_rewards_r_d) if ep_rewards_r_d else 0
        avg_r_p = np.mean(ep_rewards_r_p) if ep_rewards_r_p else 0
        avg_r_j = np.mean(ep_rewards_r_j) if ep_rewards_r_j else 0
        # --- End Calculate --- 

        # --- Calculate Action Statistics for the Episode ---
        # Combine actions across steps: (episode_len, n_agents, 1)
        all_discrete_actions = np.array(ep_discrete_actions) if ep_discrete_actions else np.empty((0, self.n_agents, 1))
        all_continuous_params = np.array(ep_continuous_params) if ep_continuous_params else np.empty((0, self.n_agents, 1))
        
        # Calculate average power used by each agent
        avg_power_per_agent = np.mean(all_continuous_params, axis=0).flatten() if all_continuous_params.size > 0 else np.zeros(self.n_agents)
        avg_power_overall = np.mean(avg_power_per_agent)
        
        # Calculate distribution of discrete actions (counts for each action type 0 to n_actions-1)
        # Flatten actions across agents and steps
        flat_discrete_actions = all_discrete_actions.flatten()
        action_counts = np.zeros(self.args.n_actions)
        if flat_discrete_actions.size > 0:
            counts = np.bincount(flat_discrete_actions.astype(int), minlength=self.args.n_actions)
            action_counts = counts[:self.args.n_actions] # Ensure correct length
        
        action_distribution = action_counts / max(1, np.sum(action_counts)) # Normalize to percentages
        # --- End Calculate Action Statistics ---

        # Logging info
        run_info = {
            "episode_length": step,
            "episode_return": episode_return,
            "avg_step_reward": avg_step_reward, # Add the average step reward
            # --- Add average reward components to run_info --- 
            "avg_r_d": avg_r_d,
            "avg_r_p": avg_r_p,
            "avg_r_j": avg_r_j,
            # --- Add action stats to run_info --- 
            "avg_power_overall": avg_power_overall,
            "action_distribution": action_distribution # Store the normalized distribution array
            # --- End Add --- 
        }
        if "individual_rewards" in info:
            run_info["individual_rewards_final"] = info["individual_rewards"]

        return run_info

    def close_env(self):
        self.env.close()

# Helper class to store data for a single episode before adding to buffer
class EpisodeBatch:
    def __init__(self, args, max_seq_length, n_agents, obs_shape):
        self.args = args
        self.max_seq_length = max_seq_length
        self.n_agents = n_agents
        self.obs_shape = obs_shape # Store obs_shape passed from runner
        self.scheme = self._get_scheme(args)
        self.data = {}
        self.t = 0

    def _get_scheme(self, args):
        env_info = args.env_info
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": self.obs_shape, "group": "agents"}, # Use stored obs_shape
            "actions_discrete": {"vshape": (1,), "group": "agents", "dtype": np.int32},
            "actions_continuous": {"vshape": (1,), "group": "agents", "dtype": np.float32},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": np.int64},
            "reward": {"vshape": (1,)}, 
            "terminated": {"vshape": (1,), "dtype": np.bool_},
            "hidden_state": {"vshape": (args.rnn_hidden_dim,), "group": "agents", "dtype": np.float32}
        }
        return scheme

    def _init_data(self):
        self.data = {}
        for key, info in self.scheme.items():
             shape = info["vshape"]
             # Ensure shape is a tuple
             if isinstance(shape, int):
                 shape = (shape,)
                 
             dtype = info.get("dtype", np.float32)
             group = info.get("group", None)
             
             # Determine required sequence length based on whether it needs T+1 steps
             # Keys needing T+1: state, obs, avail_actions, hidden_state
             requires_t_plus_1 = key in ["state", "obs", "avail_actions", "hidden_state"]
             seq_len = self.max_seq_length + 1 if requires_t_plus_1 else self.max_seq_length
             
             if group == "agents":
                 full_shape = (seq_len, self.n_agents) + shape
             else:
                 full_shape = (seq_len,) + shape
             self.data[key] = np.zeros(full_shape, dtype=dtype)
        self.t = 0

    def push(self, transition_data):
        if self.t == 0:
             self._init_data() # Initialize storage when first data comes in
             
        if self.t < self.max_seq_length:
             for key, data in transition_data.items():
                 if key in self.data:
                     self.data[key][self.t] = data
             self.t += 1
        else:
             print("Warning: Episode length exceeded max_seq_length. Data not stored.")
             
    def update_last(self, last_data):
        # Store the final observation/state needed for Q(s_T) calculation
        # These keys are often needed shifted by one in the buffer (s_T+1, obs_T+1)
        # The replay buffer sampling should handle providing s, a, r, s', term
        # Let's store the final ones at index t (which is episode_len)
        if self.t < self.max_seq_length:
             for key, data in last_data.items():
                 if key in self.data:
                      self.data[key][self.t] = data
        # else: don't store if already full

    def get_batch_data(self):
        actual_length = self.t
        batch_data = {}
        for key, info in self.scheme.items():
            # Fetch data from self.data using the key
            data_array = self.data.get(key)
            if data_array is None:
                 print(f"Warning: Key {key} not found in collected episode data.")
                 continue # Should not happen if push/init is correct
                 
            # Determine the slice length needed (T or T+1)
            requires_t_plus_1 = key in ["state", "obs", "avail_actions", "hidden_state"]
            slice_len = actual_length + 1 if requires_t_plus_1 else actual_length
            
            # Slice the data and wrap in a list for buffer compatibility
            batch_data[key] = [data_array[:slice_len]]

        # Remove keys not directly used by buffer.store_episode if necessary
        # Example: 'actions' list was added before, but maybe not needed if buffer uses discrete/continuous.
        # Assuming buffer uses discrete/continuous directly based on scheme.

        return batch_data 