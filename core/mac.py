"""
Defines the Basic Multi-Agent Controller (MAC).

This class is responsible for:
- Holding the shared agent network model (e.g., RNNAgent).
- Processing batches of observations through the agent network.
- Selecting discrete actions using an action selector (e.g., EpsilonGreedy).
- Generating associated continuous parameters (e.g., power levels) using the actor component of the agent network.
- Managing the hidden states for the recurrent part of the agent network.
"""
import torch
# from .agent import MA_CJDAgent # This was likely an older or alternative agent definition, now using RNNAgent
from .networks import RNNAgent # Import the agent network definition
from utils.action_selectors import EpsilonGreedyActionSelector # Import action selection strategy
import os
import numpy as np

class BasicMAC:
    """
    Basic Multi-Agent Controller (MAC) adapted for MP-DQN.

    Houses the shared agent network (RNNAgent with MP-DQN structure).
    Coordinates action selection using the multi-pass approach:
    1. Get updated hidden state and all continuous params from the agent.
    2. Calculate Q(s, T_i, P_i) for all discrete actions T_i using the *same* hidden state.
    3. Select discrete action T_chosen based on calculated Q-values.
    4. Return T_chosen and its corresponding continuous parameter P_chosen.
    Manages the hidden state for the RNN part of the agent network.
    """
    def __init__(self, input_shape, args):
        """
        Initializes the BasicMAC.

        Args:
            input_shape (int or tuple): The shape of the observation input for a single agent.
                                        If tuple, it's flattened.
            args (SimpleNamespace): Configuration object containing necessary parameters like
                                     n_agents, network dimensions, epsilon parameters, etc.
        """
        self.n_agents = args.n_agents
        self.args = args
        
        # Ensure input_shape is a flattened integer size
        if isinstance(input_shape, tuple):
             # Calculate the product of dimensions if input_shape is a tuple
             self.input_shape = int(np.prod(input_shape))
        else:
             self.input_shape = input_shape 

        # Build the agent network model (shared across agents)
        self._build_agents(self.input_shape, args)
        # Initialize the action selector (e.g., EpsilonGreedy)
        self.action_selector = EpsilonGreedyActionSelector(args)

        # Hidden state: Shape managed according to GRU requirements
        # Initialized externally via init_hidden
        self.hidden_states = None 

    def select_actions(self, obs_batch, avail_actions_batch, t_env, test_mode=False):
        """ 
        Selects actions (discrete and continuous) for a batch of observations using MP-DQN logic.

        Steps:
        1. Get updated hidden state and all continuous params from the agent.
        2. Calculate Q(s, T_i, P_i) for all discrete actions T_i using the *same* hidden state.
        3. Select discrete action T_chosen based on calculated Q-values.
        4. Return T_chosen and its corresponding continuous parameter P_chosen.
        
        Args:
            obs_batch (torch.Tensor): Batch of observations (batch_size, n_agents, obs_shape).
            avail_actions_batch (torch.Tensor): Batch of available actions masks 
                                              (batch_size, n_agents, n_actions).
            t_env (int): Current environment time step, used for epsilon annealing.
            test_mode (bool): If True, forces greedy action selection (disables epsilon-greedy).
            
        Returns:
            tuple: (chosen_actions_discrete, chosen_continuous_params)
                   - chosen_actions_discrete (torch.Tensor): Chosen discrete action indices 
                                                             (batch_size, n_agents, 1).
                   - chosen_continuous_params (torch.Tensor): Continuous parameters associated with 
                                                              the chosen discrete actions 
                                                              (batch_size, n_agents, 1).
        """
        # Ensure tensors and hidden states are on the correct device
        device = next(self.agent.parameters()).device
        if obs_batch.device != device:
            obs_batch = obs_batch.to(device)
        if self.hidden_states is None:
             self.init_hidden(batch_size=obs_batch.shape[0])
             # print("Warning: MAC hidden state was None in select_actions, initialized.")
        if self.hidden_states.device != device:
             self.hidden_states = self.hidden_states.to(device)
        if avail_actions_batch.device != device:
            avail_actions_batch = avail_actions_batch.to(device)

        # --- Step 1: Get Updated Hidden State and All Continuous Params ---
        # Reshape observations for agent input: (batch * n_agents, obs_shape)
        batch_size = obs_batch.shape[0]
        obs_reshaped = obs_batch.reshape(-1, self.input_shape)

        # Get updated hidden state and all continuous params P for each action T_i
        # hidden_states_updated shape: (batch * n_agents, rnn_hidden_dim)
        # continuous_params_all shape: (batch * n_agents, n_actions)
        hidden_states_updated, continuous_params_all = self.forward(obs_reshaped, self.hidden_states)

        # Update the stored hidden state for the next step
        self.hidden_states = hidden_states_updated.detach() # Detach to prevent gradients flowing back from next step

        # --- Step 2: Calculate Q(s, T_i, P_i) for all discrete actions T_i --- 
        # Initialize tensor to store Q-values for all actions
        # Shape: (batch * n_agents, n_actions)
        all_q_values = torch.zeros_like(continuous_params_all)

        # Loop through each possible discrete action T_i
        for action_idx in range(self.args.n_actions):
            # Create tensor for the current discrete action index
            # Shape: (batch * n_agents, 1)
            current_batch_x_agents = hidden_states_updated.shape[0] # Should be batch_size * n_agents
            current_action_idx_tensor = torch.full((current_batch_x_agents, 1),
                                                         fill_value=action_idx,
                                                         dtype=torch.long, device=device)

            # Get the continuous parameter P_i corresponding to this action_idx
            # Shape: (batch * n_agents, 1)
            current_param = continuous_params_all[:, action_idx].unsqueeze(1)

            # Calculate Q-value for this specific (state, T_i, P_i) tuple
            # Uses the *updated* hidden state from step 1
            # Shape: (batch * n_agents, 1)
            q_value = self.agent.get_q_value_for_action(hidden_states_updated,
                                                          current_action_idx_tensor,
                                                          current_param)

            # Store the calculated Q-value in the corresponding column
            all_q_values[:, action_idx] = q_value.squeeze(1)

        # Reshape Q-values back to (batch_size, n_agents, n_actions)
        agent_qs = all_q_values.view(batch_size, self.n_agents, -1)

        # --- Step 3: Select Discrete Actions --- 
        # Apply availability mask
        agent_qs[avail_actions_batch == 0] = -float("Inf")
        chosen_actions_discrete = self.action_selector.select_action(agent_qs, 
                                                                    avail_actions_batch, # Pass avail actions again just in case selector uses it
                                                                    t_env, 
                                                                    test_mode=test_mode)
        # chosen_actions_discrete shape: (batch_size, n_agents, 1)

        # --- Step 4: Gather Continuous Parameters for Chosen Actions ---
        # Reshape chosen actions to flat: (batch*n_agents, 1)
        chosen_actions_discrete_flat = chosen_actions_discrete.view(-1, 1).long()
        
        # Gather the corresponding parameters from the actor's output
        # Ensure indices are on the same device
        if chosen_actions_discrete_flat.device != continuous_params_all.device:
            chosen_actions_discrete_flat = chosen_actions_discrete_flat.to(continuous_params_all.device)

        chosen_continuous_params_flat = torch.gather(continuous_params_all,
                                                     dim=1, 
                                                     index=chosen_actions_discrete_flat)
        # chosen_continuous_params_flat shape: (batch*n_agents, 1)
        
        # Reshape gathered params back to (batch_size, n_agents, 1)
        chosen_continuous_params = chosen_continuous_params_flat.view(batch_size, self.n_agents, 1)

        return chosen_actions_discrete, chosen_continuous_params

    def forward(self, agent_inputs_reshaped, hidden_states):
        """
        Processes observations to get updated hidden state and all continuous parameters.

        Args:
            agent_inputs_reshaped (torch.Tensor): Shape: (batch_size * n_agents, obs_shape).
            hidden_states (torch.Tensor): Shape: (batch_size * n_agents, rnn_hidden_dim).

        Returns:
            tuple: (h_out, continuous_params_all)
                   h_out shape: (batch_size * n_agents, rnn_hidden_dim)
                   continuous_params_all shape: (batch_size * n_agents, n_actions)
        """
        # Get updated hidden state from RNN
        h_out = self.agent.forward(agent_inputs_reshaped, hidden_states)

        # Get all continuous parameters from Actor
        continuous_params_all = self.agent.actor_forward(agent_inputs_reshaped)

        return h_out, continuous_params_all

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state for the MAC (for nn.GRUCell).
        """
        # For GRUCell, the agent expects hidden state shape: (batch * n_agents, hidden_dim)
        hidden_dim = self.args.rnn_hidden_dim
        self.hidden_states = self.agent.init_hidden().repeat(batch_size * self.n_agents, 1)
        # Ensure it's on the correct device (using agent's parameter device as reference)
        device = next(self.agent.parameters()).device
        self.hidden_states = self.hidden_states.to(device)

    # --- Utility methods for interacting with the internal agent network --- 

    def parameters(self):
        """ Returns the parameters of the internal agent network. Used by the optimizer. """
        return self.agent.parameters()

    def load_state(self, other_mac_state_dict):
        """ 
        Loads the state dictionary into the internal agent network.
        Used primarily for updating the target MAC's agent network from the evaluation MAC.
        
        Args:
            other_mac_state_dict (dict): State dictionary compatible with the internal agent network (RNNAgent).
        """
        self.agent.load_state_dict(other_mac_state_dict)

    def state_dict(self):
         """ Returns the state dictionary of the internal agent network. Used for saving the evaluation MAC. """
         return self.agent.state_dict()

    def cuda(self):
        """ Moves the internal agent network to the GPU (CUDA device). """
        self.agent.cuda()

    def save_models(self, path):
        """
        Saves the state dictionary of the internal agent network to a file.
        
        Args:
            path (str): Directory path where the model file (`agent.pth`) will be saved.
        """
        os.makedirs(path, exist_ok=True) # Ensure the directory exists
        # Save the agent network's state dictionary
        torch.save(self.agent.state_dict(), f"{path}/agent.pth")

    def load_models(self, path):
        """
        Loads the state dictionary into the internal agent network from a specified file.

        Args:
            path (str): Directory path containing the model file (`agent.pth`).
        """
        # Determine the device the agent network is currently supposed to be on
        device = next(self.agent.parameters()).device 
        # Load the state dict, ensuring it's mapped to the agent's current device 
        # (e.g., loads correctly whether saved from CPU or GPU, to CPU or GPU)
        self.agent.load_state_dict(torch.load(f"{path}/agent.pth", map_location=device))

    def _build_agents(self, input_shape, args):
        """ 
        Protected helper method to instantiate the agent network. 
        Currently hardcoded to use RNNAgent.
        """
        # RNNAgent includes both the RNN component (for Q-values) and the Actor component (for continuous params).
        self.agent = RNNAgent(input_shape, args)

    # ---------------------------------------------------------------------------
    # Unused/Helper methods (Potentially from older versions or alternative designs)
    # Kept commented out for reference if input processing logic needs changes.
    # ---------------------------------------------------------------------------
    # def _build_inputs(self, batch, t):
    #     ...
    # def _get_input_shape(self, scheme):
    #     ...
    # ---------------------------------------------------------------------------

    # Need an action selector (e.g., epsilon-greedy)
    # This is often managed separately or within the runner.
    # Let's assume the runner handles action selection logic for now.
    # The `select_actions` method above needs refinement based on how the runner integrates.
    # For now, this MAC focuses on forwarding observations through the shared agent network. 