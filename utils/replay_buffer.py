"""
Implementation of an Episode Replay Buffer.

Stores entire episodes of experience (state, obs, actions, reward, etc.) 
for use in off-policy MARL algorithms like QMix.
It handles episodes of varying lengths by padding up to a maximum episode limit.
Provides methods for storing episodes and sampling batches of episodes.
"""
import numpy as np
import threading # For thread-safe operations if used in parallel environments

class EpisodeReplayBuffer:
    """
    Stores complete episodes for MARL training, particularly suitable for QMix.
    
    Handles padding for episodes of varying lengths up to `episode_limit`.
    Stores transitions including global state, individual observations, discrete actions, 
    continuous action parameters, available actions masks, shared rewards, and termination flags.
    Uses NumPy arrays for efficient storage.
    Provides thread-safe methods for storing and sampling.
    """
    def __init__(self, args):
        """
        Initializes the Episode Replay Buffer.

        Args:
            args (SimpleNamespace): Configuration object containing buffer parameters:
                - buffer_size (int): Maximum number of episodes to store.
                - episode_limit (int): Maximum length of a single episode (used for padding).
                - n_actions (int): Number of discrete actions available to each agent.
                - n_agents (int): Number of agents in the environment.
                - state_shape (int or tuple): Shape of the global state.
                - obs_shape (int or tuple): Shape of the local observation for a single agent.
        """
        self.args = args
        self.buffer_size = args.buffer_size       # Max episodes
        self.episode_limit = args.episode_limit # Max length T
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        # Ensure shapes are integers (flattened size)
        self.state_shape = int(np.prod(args.state_shape)) if isinstance(args.state_shape, tuple) else args.state_shape
        self.obs_shape = int(np.prod(args.obs_shape)) if isinstance(args.obs_shape, tuple) else args.obs_shape

        # --- Buffer Storage Initialization ---
        # Dimensions: (buffer_capacity, max_episode_length + 1, ...) for state/obs/avail_actions
        # Dimensions: (buffer_capacity, max_episode_length, ...) for actions/reward/terminated/filled
        # We store T+1 steps for state, obs, avail_actions to easily get the next state/obs/actions 
        # required for TD-learning targets.
        self.buffers = {
            # Global state: T+1 steps
            'state': np.empty((self.buffer_size, self.episode_limit + 1, self.state_shape), dtype=np.float32),
            # Agent observations: T+1 steps
            'obs': np.empty((self.buffer_size, self.episode_limit + 1, self.n_agents, self.obs_shape), dtype=np.float32),
            # Discrete actions taken: T steps
            'actions_discrete': np.empty((self.buffer_size, self.episode_limit, self.n_agents, 1), dtype=np.int32),
            # Continuous parameters associated with actions: T steps (assuming 1 parameter per agent)
            'actions_continuous': np.empty((self.buffer_size, self.episode_limit, self.n_agents, 1), dtype=np.float32), 
            # Available actions mask: T+1 steps
            'avail_actions': np.empty((self.buffer_size, self.episode_limit + 1, self.n_agents, self.n_actions), dtype=np.int64),
            # Shared reward: T steps
            'reward': np.empty((self.buffer_size, self.episode_limit, 1), dtype=np.float32),
            # Termination flag (episode ended): T steps
            'terminated': np.empty((self.buffer_size, self.episode_limit, 1), dtype=np.bool_),
            # Mask indicating valid (non-padded) steps within an episode: T steps
            'filled': np.empty((self.buffer_size, self.episode_limit, 1), dtype=np.bool_),
            # ADD HIDDEN STATE BUFFER
            'hidden_state': np.empty((self.buffer_size, self.episode_limit + 1, self.n_agents, args.rnn_hidden_dim), dtype=np.float32)
        }

        # --- Tracking Variables ---
        self.current_index = 0 # Next index to write to (circular buffer logic)
        self.current_size = 0  # Number of episodes currently stored
        self.lock = threading.Lock() # Ensures thread-safety during storage/sampling if needed

        print(f"Replay Buffer Initialized: Size={self.buffer_size}, Episode Limit={self.episode_limit}")
        # print(f"  State shape: {self.state_shape}, Obs shape: {self.obs_shape}") # Optional debug print

    def store_episode(self, episode_batch):
        """
        Stores a completed episode batch into the buffer.
        
        Handles padding for episodes shorter than `episode_limit`.
        Assumes `episode_batch` contains transitions for a single episode, typically
        provided by an `EpisodeRunner`.

        Args:
            episode_batch (dict): A dictionary where keys are data types ('state', 'obs', 'actions', etc.)
                                  and values are lists containing a single NumPy array representing the 
                                  data for one full episode. 
                                  'actions' is expected to be a list of tuples [(d1, c1), (d2, c2), ...].
                                  Lengths: state, obs, avail_actions are T+1. Others are T.
        """
        # The runner typically wraps the episode data in a list of length 1.
        batch_size = len(episode_batch['state']) 
        if batch_size != 1:
            # This implementation assumes one episode is stored at a time.
            # Could be adapted for multi-episode storage if the runner changes.
            print("Warning: EpisodeReplayBuffer expects batch_size=1 from runner")
            # For now, proceed assuming the first element is the episode.

        # Ensure thread-safe access to buffer indices and data
        with self.lock:
            # Get the index in the buffer where this episode will be stored
            storage_indices = self._get_storage_idx(inc=batch_size)
            idx = storage_indices[0] # Since batch_size is assumed 1

            # Extract the actual episode data (NumPy arrays) from the wrapping list
            ep_data = {k: v[0] for k, v in episode_batch.items()}
            
            # Determine actual episode length (e.g., from reward array)
            actual_episode_len = ep_data['reward'].shape[0] 
            
            # --- Store Data with Padding --- 
            # Store data with T+1 length (state, obs, avail_actions, hidden_state)
            self.buffers['state'][idx, :actual_episode_len + 1] = ep_data['state']
            self.buffers['obs'][idx, :actual_episode_len + 1] = ep_data['obs']
            self.buffers['avail_actions'][idx, :actual_episode_len + 1] = ep_data['avail_actions']
            # ADD storing hidden_state
            if 'hidden_state' in ep_data: # Check if hidden_state is provided
                self.buffers['hidden_state'][idx, :actual_episode_len + 1] = ep_data['hidden_state']
            else:
                # Optional: Handle case where hidden state wasn't provided (e.g., fill with zeros or raise error)
                print("Warning: 'hidden_state' key not found in episode_batch data during buffer storage.")
                self.buffers['hidden_state'][idx, :actual_episode_len + 1] = 0 # Fill with zeros as fallback

            # Store data with T length (actions, reward, terminated)
            self.buffers['actions_discrete'][idx, :actual_episode_len] = ep_data['actions_discrete']
            self.buffers['actions_continuous'][idx, :actual_episode_len] = ep_data['actions_continuous']
            self.buffers['reward'][idx, :actual_episode_len] = ep_data['reward']
            self.buffers['terminated'][idx, :actual_episode_len] = ep_data['terminated']

            # Create and store the 'filled' mask (length T)
            self.buffers['filled'][idx] = False # Initialize all to False
            self.buffers['filled'][idx, :actual_episode_len] = True # Mark actual steps as True

            # --- Pad Remaining Steps --- 
            # Pad steps beyond the actual episode length up to episode_limit
            # Pad T+1 length data (from T+1 onwards)
            self.buffers['state'][idx, actual_episode_len + 1:] = 0
            self.buffers['obs'][idx, actual_episode_len + 1:] = 0
            self.buffers['avail_actions'][idx, actual_episode_len + 1:] = 0 # Assuming 0 means unavailable
            # ADD padding hidden_state
            self.buffers['hidden_state'][idx, actual_episode_len + 1:] = 0
            
            # Pad T length data (from T onwards)
            self.buffers['actions_discrete'][idx, actual_episode_len:] = 0 
            self.buffers['actions_continuous'][idx, actual_episode_len:] = 0.0
            self.buffers['reward'][idx, actual_episode_len:] = 0
            # Mark padded steps as terminated (important for TD target calculation)
            self.buffers['terminated'][idx, actual_episode_len:] = True
            # Filled mask already handled above (padded steps remain False)

    def sample(self, batch_size):
        """
        Samples a batch of complete episodes from the buffer.

        Args:
            batch_size (int): The number of episodes to sample.

        Returns:
            dict or None: A dictionary containing batches of episode data, sliced to the 
                          maximum actual sequence length (`max_seq_len`) found in the sampled batch. 
                          Includes a 'max_seq_len' key indicating this length (T).
                          Returns None if the buffer is empty or `batch_size` is invalid.
        """
        # Prevent sampling if buffer has fewer episodes than requested (or is empty)
        if self.current_size < batch_size:
             print(f"Warning: Sampling {batch_size} but buffer only contains {self.current_size} episodes. Sampling {self.current_size}.")
             actual_batch_size = self.current_size
        else:
             actual_batch_size = batch_size
             
        if actual_batch_size <= 0:
            print("Error: Cannot sample 0 or negative episodes.")
            return None 

        # Randomly select episode indices without replacement
        indices = np.random.choice(self.current_size, actual_batch_size, replace=False)

        # --- Gather Data and Find Max Sequence Length --- 
        sampled_batch = {}
        max_seq_len = 0 # Track the longest actual episode length (T) in the batch
        for key in self.buffers:
            sampled_batch[key] = self.buffers[key][indices]
            # Find the maximum actual length (T) in the sampled batch using the 'filled' mask
            if key == 'filled':
                # Sum the boolean mask along the time dimension (axis 1) for each episode
                # The max of these sums gives the length of the longest episode in the batch
                episode_lengths = np.sum(sampled_batch[key], axis=1).flatten()
                if len(episode_lengths) > 0: # Handle case where batch might be empty (shouldn't happen with checks above)
                     max_seq_len = int(np.max(episode_lengths))
                else:
                     max_seq_len = 0 # Should not be reachable

        # --- Truncate Sequences to Max Length --- 
        # Slice the sampled data arrays to the determined max_seq_len (T)
        # Arrays with T+1 length are sliced to max_seq_len + 1
        # Arrays with T length are sliced to max_seq_len
        final_batch = {}
        for key, data in sampled_batch.items():
             if key in ['state', 'obs', 'avail_actions', 'hidden_state']:
                 # Slice up to T+1 (exclusive index max_seq_len + 1)
                 final_batch[key] = data[:, :max_seq_len + 1]
             elif key in ['actions_discrete', 'actions_continuous', 'reward', 'terminated', 'filled']:
                 # Slice up to T (exclusive index max_seq_len)
                 final_batch[key] = data[:, :max_seq_len]
             else: # Should not happen if buffer keys are correct
                 print(f"Warning: Unexpected key '{key}' encountered during sampling.")
                 final_batch[key] = data 
                 
        # Add the calculated maximum sequence length (T) to the batch dictionary
        final_batch['max_seq_len'] = max_seq_len 

        return final_batch

    def _get_storage_idx(self, inc=None):
        """
        Calculates the next available index/indices for storing data using circular logic.
        Updates `self.current_index` and `self.current_size`.
        Protected method, assumes lock is acquired by caller.

        Args:
            inc (int, optional): Number of indices to request. Defaults to 1.

        Returns:
            np.ndarray: An array of indices where new data should be stored.
        """
        inc = inc or 1
        if self.current_index + inc <= self.buffer_size:
            # Enough space available consecutively
            idx = np.arange(self.current_index, self.current_index + inc)
            self.current_index += inc
        elif inc <= self.buffer_size: 
            # Need to wrap around the buffer
            # Calculate how many indices overflow to the beginning
            overflow = inc - (self.buffer_size - self.current_index)
            # Get indices from current position to the end
            idx_a = np.arange(self.current_index, self.buffer_size)
            # Get indices from the beginning for the overflow
            idx_b = np.arange(0, overflow)
            # Concatenate the two parts
            idx = np.concatenate((idx_a, idx_b))
            # Update current index to the position after the wrapped-around part
            self.current_index = overflow
        else:
            # This should not happen if buffer_size is respected
            raise ValueError("Attempting to store more episodes than the buffer capacity in a single call.")
            
        # Update the current number of episodes stored, capped by buffer size
        self.current_size = min(self.current_size + inc, self.buffer_size)
        return idx

    def __len__(self):
        """ Returns the current number of episodes stored in the buffer. """
        # Acquire lock for thread-safe access to current_size
        with self.lock:
             return self.current_size 