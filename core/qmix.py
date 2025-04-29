"""
Defines the QMix Learner.

This class implements the core QMix algorithm logic:
- Manages the agent networks (via the Multi-Agent Controller, MAC), the mixing network, 
  and their corresponding target networks.
- Handles the optimization process (using an Adam optimizer).
- Calculates the QMix loss based on sampled batches from the replay buffer.
- Performs training steps to update network parameters.
- Periodically updates the target networks.
- Provides methods for saving and loading model checkpoints.

Reference: QMix: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning (Rashid et al., 2018)
https://arxiv.org/abs/1803.11485
"""
import torch
import torch.nn as nn
import torch.optim as optim
import copy # For deep copying networks (target networks)
import numpy as np
import os # Import the os module for file system operations

from .networks import RNNAgent, QMixer # Import network architectures

class QMixLearner:
    """
    Manages the training process for QMix, adapted for MP-DQN.

    Contains the evaluation and target networks (agents via MAC, mixer),
    the optimizer, and the main training logic for updating these networks based on
    batches of experience sampled from the replay buffer.
    Requires MAC and Agent networks implementing the MP-DQN structure.
    """
    def __init__(self, mac, args):
        """
        Initializes the QMixLearner.

        Args:
            mac (BasicMAC): The Multi-Agent Controller containing the shared evaluation agent network.
            args (SimpleNamespace): Configuration object with parameters like learning rate, 
                                    discount factor (gamma), network dimensions, target update interval, etc.
        """
        self.args = args
        self.mac = mac # The MAC holds the evaluation agent network(s)
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.device = torch.device(args.device if torch.cuda.is_available() and args.use_cuda else "cpu")

        # --- Initialize Networks ---
        self.eval_qmix_net = QMixer(args) # Mixing network
        self.target_mac = copy.deepcopy(mac)
        self.target_qmix_net = QMixer(args)
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        # Move networks to the designated device (CPU or CUDA)
        if args.use_cuda:
            self.mac.cuda() # Ensure eval MAC agent is on CUDA
            self.target_mac.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
        
        print(f"QMix Learner Initialized on device: {self.device}")

        # --- Optimizer Setup ---
        self.agent_params = list(self.mac.parameters())
        self.qmix_params = list(self.eval_qmix_net.parameters())
        self.params = self.agent_params + self.qmix_params
        self.optimizer = optim.Adam(params=self.params, lr=args.lr)

        # --- Training State Tracking ---
        self.last_target_update_step = 0
        self.train_step = 0

    def train(self, batch, train_info):
        """
        Performs a single QMix training update step using MP-DQN logic.

        Calculates the TD-error based loss between the Q_tot predicted by the evaluation 
        networks and the target Q_tot derived from the target networks, then performs 
        a gradient descent step.

        Args:
            batch (dict): A dictionary containing a batch of episode transitions sampled 
                          from the replay buffer. Keys include 'state', 'obs', 
                          'actions_discrete', 'actions_continuous', 'reward', 'terminated', 'filled'.
            train_info (dict): Dictionary possibly containing additional training 
                               information (e.g., current total_steps, although not used here).

        Returns:
            dict: A dictionary containing the calculated loss value and other statistics for this training step.
        """
        self.train_step += 1
        
        # --- Prepare Batch Data ---
        # Extract data components from the batch dictionary
        states = torch.tensor(batch['state'], dtype=torch.float32).to(self.device)
        obs = torch.tensor(batch['obs'], dtype=torch.float32).to(self.device)
        # Discrete actions taken by agents
        actions_discrete = torch.tensor(batch['actions_discrete'], dtype=torch.long).to(self.device)
        # Continuous parameters associated with the discrete actions taken
        actions_continuous = torch.tensor(batch['actions_continuous'], dtype=torch.float32).to(self.device)
        # Global reward signal
        rewards = torch.tensor(batch['reward'], dtype=torch.float32).to(self.device)
        # Termination status (True if episode ended at this step)
        terminated = torch.tensor(batch['terminated'], dtype=torch.bool).to(self.device)
        # Mask indicating valid (non-padded) steps in the episode sequences
        mask = torch.tensor(batch['filled'], dtype=torch.float32).squeeze(-1).to(self.device)
        hidden_states = torch.tensor(batch['hidden_state'], dtype=torch.float32).to(self.device)
        # Maximum sequence length in this batch (episodes might have different lengths)
        max_seq_len = batch['max_seq_len']
        
        # Trim sequences to the maximum length in the batch
        # Shapes: (batch_size, max_seq_len, ...)
        batch_size = states.shape[0]
        states = states[:, :max_seq_len]
        obs = obs[:, :max_seq_len]
        actions_discrete = actions_discrete[:, :max_seq_len]
        actions_continuous = actions_continuous[:, :max_seq_len]
        rewards = rewards[:, :max_seq_len]
        terminated = terminated[:, :max_seq_len]
        mask = mask[:, :max_seq_len]
        hidden_states = hidden_states[:, :max_seq_len+1]

        # --- Calculate Target Q-values using Target Networks (Double Q-Learning + MP-DQN) ---
        # 1. Get MP-DQN Q-values (Q(s', T_i, P_i)) for ALL actions T_i using the *target* MAC/Agent.
        #    Also get all continuous params P_i from the target Actor.
        target_q_vals_all_actions, _ = self._get_all_action_q_values_and_params(self.target_mac, batch, max_seq_len)
        # target_q_vals_all_actions shape: (batch_size, max_seq_len, n_agents, n_actions)

        # 2. Get MP-DQN Q-values (Q(s', T_i, P_i)) for ALL actions T_i using the *evaluation* MAC/Agent.
        #    This is needed to select the best action for the next state (Double DQN part).
        eval_q_vals_all_actions, _ = self._get_all_action_q_values_and_params(self.mac, batch, max_seq_len)
        # eval_q_vals_all_actions shape: (batch_size, max_seq_len, n_agents, n_actions)

        # Select best actions for the next state (s_t+1) using the *evaluation* Q-values
        eval_q_vals_next = eval_q_vals_all_actions[:, 1:]
        # Mask unavailable actions in the next state before taking argmax
        # Assuming avail_actions are also in the batch, otherwise need to fetch/compute
        # avail_actions = torch.tensor(batch['avail_actions'], dtype=torch.float32).to(self.device)[:, 1:]
        # eval_q_vals_next[avail_actions == 0] = -float("Inf") # Apply mask if avail_actions provided
        next_actions_discrete = eval_q_vals_next.argmax(dim=3, keepdim=True) # Shape: (batch_size, max_seq_len-1, n_agents, 1)

        # 3. Get the *target* network's Q-values for these chosen next actions.
        # We gather from target_q_vals_all_actions using the indices selected by the eval network.
        target_q_taken = torch.gather(target_q_vals_all_actions[:, 1:], dim=3, index=next_actions_discrete).squeeze(3)
        # target_q_taken shape: (batch_size, max_seq_len-1, n_agents)

        # 4. Calculate the target Q_tot using the target mixer network.
        target_q_mixer = self.target_qmix_net(target_q_taken, states[:, 1:])
        # target_q_mixer shape: (batch_size, max_seq_len-1, 1)

        # 5. Calculate the final TD target: y = r + gamma * target_Q_tot(s', a')
        targets = rewards[:, :-1] + self.args.gamma * (1 - terminated[:, :-1].float()) * target_q_mixer
        # targets shape: (batch_size, max_seq_len-1, 1)

        # --- Calculate Evaluation Q-values Q(s_t, a_t) using buffer data ---
        # Get data for time steps t = 0 to max_seq_len-1
        # Correction: We need states/actions/hidden_states for t=0 to max_seq_len-2 for loss calculation
        hidden_states_t = hidden_states[:, :max_seq_len-1] # Shape: (batch, max_seq_len-1, n_agents, hidden_dim) CORRECTED SLICE
        actions_discrete_t = actions_discrete[:, :max_seq_len-1] # Shape: (batch, max_seq_len-1, n_agents, 1) CORRECTED SLICE (if needed, was [:,:-1])
        actions_continuous_t = actions_continuous[:, :max_seq_len-1] # Shape: (batch, max_seq_len-1, n_agents, 1) CORRECTED SLICE (if needed, was [:,:-1])

        # Reshape for agent input: (batch * (max_seq_len-1) * n_agents, ...)
        # Calculate effective batch size, handling potential zero length sequences
        if max_seq_len <= 1:
             # If sequence length is 1 or less, q_taken will be empty
             q_taken = torch.empty((batch_size, 0, self.n_agents), device=self.device)
        else:
            batch_size_eff = batch_size * (max_seq_len - 1) * self.n_agents
            hidden_states_t_flat = hidden_states_t.reshape(batch_size_eff, self.args.rnn_hidden_dim)
            actions_discrete_t_flat = actions_discrete_t.reshape(batch_size_eff, 1)
            actions_continuous_t_flat = actions_continuous_t.reshape(batch_size_eff, 1)

            # Calculate Q(s_t, T_t, P_t) using the evaluation MAC's agent
            # Call get_q_value_for_action directly with data from the buffer
            q_taken_flat = self.mac.agent.get_q_value_for_action(hidden_states_t_flat,
                                                                actions_discrete_t_flat,
                                                                actions_continuous_t_flat)
            # q_taken_flat shape: (batch * (max_seq_len-1) * n_agents, 1)

            # Reshape back to (batch_size, max_seq_len-1, n_agents)
            q_taken = q_taken_flat.view(batch_size, max_seq_len - 1, self.n_agents)

        # 2. Mix the evaluation Q-values using the evaluation mixer network.
        eval_q_mixer = self.eval_qmix_net(q_taken, states[:, :-1])
        # eval_q_mixer shape: (batch_size, max_seq_len-1, 1)

        # --- Calculate Loss ---
        td_error = (eval_q_mixer - targets.detach())
        mask = mask[:, :-1]
        masked_td_error = td_error * mask.unsqueeze(-1)
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # --- Optimization Step ---
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimizer.step()

        # --- Update Target Networks ---
        if (self.train_step - self.last_target_update_step) >= self.args.target_update_interval:
            self._update_targets()
            self.last_target_update_step = self.train_step
            # print(f"Step {self.train_step}: Updated target networks.") # Optional debug print

        # Return the loss value and other stats for logging
        stats = {
            'loss': loss.item(),
            'grad_norm': grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, # Handle tensor/float case
            'eval_qtot_avg': eval_q_mixer.mean().item(),
            'target_qtot_avg': targets.mean().item()
        }
        return stats

    def _get_all_action_q_values_and_params(self, mac_controller, batch, max_seq_len):
        """
        Calculates Q-values Q(s, T_i, P_i) for ALL discrete actions T_i for a given batch
        using the MP-DQN MAC and Agent.

        Args:
            mac_controller: The MAC instance (eval or target).
            batch: The sampled batch dictionary.
            max_seq_len: The maximum sequence length T.

        Returns:
            torch.Tensor: Q-values for all actions, shape (batch_size, max_seq_len, n_agents, n_actions).
            torch.Tensor: Continuous parameters for all actions, shape (batch_size, max_seq_len, n_agents, n_actions).
        """
        batch_size = batch["state"].shape[0]
        n_agents = self.args.n_agents
        n_actions = self.args.n_actions
        device = self.device # Use the learner's device

        # Prepare tensors to store results
        all_q_values_out = torch.zeros((batch_size, max_seq_len, n_agents, n_actions), device=device)
        all_params_out = torch.zeros((batch_size, max_seq_len, n_agents, n_actions), device=device)

        # Initialize hidden state for the batch
        mac_controller.init_hidden(batch_size)

        for t in range(max_seq_len):
            obs_t = torch.tensor(batch["obs"][:, t], dtype=torch.float32).to(device)
            obs_reshaped = obs_t.reshape(-1, self.obs_shape) # Shape: (batch*n_agents, obs_shape)
            hidden_state_t = mac_controller.hidden_states # Shape: (batch*n_agents, rnn_hidden_dim)

            # --- Get updated hidden state and all continuous params --- 
            # Note: mac_controller.forward now takes reshaped obs
            hidden_state_updated, continuous_params_all_flat = mac_controller.forward(obs_reshaped, hidden_state_t)
            # hidden_state_updated shape: (batch*n_agents, rnn_hidden_dim)
            # continuous_params_all_flat shape: (batch*n_agents, n_actions)
            
            # Update the controller's hidden state for the next iteration (if processing sequences)
            mac_controller.hidden_states = hidden_state_updated 

            # --- Calculate Q(s, T_i, P_i) for all T_i using the *updated* hidden state --- 
            q_values_t_flat = torch.zeros_like(continuous_params_all_flat) # (batch*n_agents, n_actions)

            for action_idx in range(n_actions):
                # Create tensor for the current discrete action index
                current_action_idx_tensor = torch.full((batch_size * n_agents, 1),
                                                       fill_value=action_idx,
                                                       dtype=torch.long, device=device)

                # Get the continuous parameter P_i corresponding to this action_idx
                current_param = continuous_params_all_flat[:, action_idx].unsqueeze(1)

                # Calculate Q-value Q(s_t, T_i, P_i) using the updated hidden state
                q_value = mac_controller.agent.get_q_value_for_action(hidden_state_updated,
                                                                      current_action_idx_tensor,
                                                                      current_param)
                # Store the calculated Q-value
                q_values_t_flat[:, action_idx] = q_value.squeeze(1)

            # Reshape and store results for this time step t
            all_q_values_out[:, t] = q_values_t_flat.view(batch_size, n_agents, n_actions)
            all_params_out[:, t] = continuous_params_all_flat.view(batch_size, n_agents, n_actions)

        return all_q_values_out, all_params_out

    def _update_targets(self):
        """ 
        Update the target network parameters by copying the parameters 
        from the evaluation networks (soft update is not used here).
        """
        # Update target MAC's agent network parameters
        self.target_mac.load_state(self.mac.state_dict())
        # Update target mixer network parameters
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def cuda(self):
        """ Moves all managed networks (eval and target) to the CUDA device. """
        self.mac.cuda()
        self.target_mac.cuda()
        self.eval_qmix_net.cuda()
        self.target_qmix_net.cuda()
        self.device = torch.device("cuda") # Update device attribute

    def save_models(self, path):
        """
        Saves the state dictionaries of the evaluation networks (agent via MAC, mixer)
        and the optimizer state.

        Args:
            path (str): Directory path to save the model files.
        """
        # Ensure the directory exists (mac.save_models also does this)
        os.makedirs(path, exist_ok=True)
        # Save agent network state (calls BasicMAC.save_models)
        self.mac.save_models(path) 
        # Save mixer network state
        torch.save(self.eval_qmix_net.state_dict(), f"{path}/qmix_net.pth")
        # Save optimizer state (useful for resuming training)
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pth")

    def load_models(self, path):
        """
        Loads the state dictionaries of the evaluation networks (agent via MAC, mixer)
        from saved files. Does NOT load optimizer state currently.

        Args:
            path (str): Directory path containing the saved model files.
        """
        # Load agent network state (calls BasicMAC.load_models)
        self.mac.load_models(path)
        # Load mixer network state, ensuring it's mapped to the correct device
        self.eval_qmix_net.load_state_dict(torch.load(f"{path}/qmix_net.pth", map_location=lambda storage, loc: storage))
        
        # NOTE: Optimizer state is not loaded here. If resuming training, 
        # you might want to load self.optimizer.load_state_dict(...) as well.
        
        # Crucially, after loading evaluation networks, update the target networks
        self._update_targets() 