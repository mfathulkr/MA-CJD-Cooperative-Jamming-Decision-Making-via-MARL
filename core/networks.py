"""
Neural network architectures used in the MA-CJD project.

Contains:
- RNNAgent: A recurrent agent network combining an RNN (GRU) for temporal 
            processing and Q-value estimation, and a feed-forward actor 
            network for generating continuous action parameters.
- QMixer: The mixing network used by the QMix algorithm, which combines 
          individual agent Q-values into a global Q_tot based on the state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RNNAgent(nn.Module):
    """
    Recurrent Agent Network intended to implement the MP-DQN structure.

    It separates Actor and Q-network logic.
    - Actor (`actor_forward`): Generates continuous parameters for all discrete actions.
    - RNN (`forward`): Processes observation to update the hidden state.
    - Q-Network (`get_q_value_for_action`): Calculates Q-value for a *specific* state,
      discrete action, continuous parameter, using the updated hidden state.

    **Note:** This structure requires the controller (MAC) to call `forward` once per step
    to get the updated hidden state, call `actor_forward` to get all parameters, and then
    call `get_q_value_for_action` multiple times (once per discrete action) using the
    *same* updated hidden state.
    """
    def __init__(self, input_shape, args):
        """
        Initializes the RNNAgent network.

        Args:
            input_shape (int): The flattened size of the observation input for a single agent.
            args (SimpleNamespace): Configuration object containing network hyperparameters
                                     (rnn_hidden_dim, actor_hidden_dim, n_actions).
        """
        super(RNNAgent, self).__init__()
        self.args = args
        self.n_actions = args.n_actions # Number of discrete actions
        self.input_shape = input_shape
        self.rnn_hidden_dim = args.rnn_hidden_dim

        # --- Actor Network Component ---
        # Input: Observation (flattened)
        # Output: Continuous parameters (e.g., power levels), one for each discrete action.
        self.actor_input_dim = input_shape
        self.actor_hidden_dim = args.actor_hidden_dim 
        self.actor_output_dim = self.n_actions # Output one parameter per discrete action
        
        # Define the sequential feed-forward actor network
        self.actor = nn.Sequential(
            nn.Linear(self.actor_input_dim, self.actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.actor_hidden_dim, self.actor_hidden_dim), # Extra hidden layer
            nn.ReLU(),
            nn.Linear(self.actor_hidden_dim, self.actor_output_dim),
            nn.Sigmoid() # Output normalized parameters (e.g., power in [0, 1])
        )

        # --- Shared RNN Component (Processes Observation for Hidden State) ---
        # Structure: Input -> FC -> ReLU -> GRUCell
        self.fc1 = nn.Linear(self.input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        
        # --- Q-Value Head Component (Uses Hidden State, Action, and Param) ---
        # Input: Concatenation of GRU hidden state, one-hot encoded discrete action,
        #        and the corresponding continuous parameter.
        # Output: Single Q-value for the specific state-action-parameter tuple.
        # Input dimension calculation:
        q_head_input_dim = self.rnn_hidden_dim + self.n_actions + 1 # hidden_state + one_hot_action + continuous_param
        # Using a simple MLP for the Q-head for now
        self.fc2_q_head = nn.Sequential(
            nn.Linear(q_head_input_dim, self.rnn_hidden_dim), # Hidden layer in Q-head
            nn.ReLU(),
            nn.Linear(self.rnn_hidden_dim, 1) # Output a single Q-value
        )

    def init_hidden(self):
        """
        Initializes the hidden state for the GRU cell.
        Returns: torch.Tensor of zeros with shape (1, rnn_hidden_dim).
        """
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, agent_inputs, h_in):
        """
        Processes observation through RNN to update hidden state.

        Args:
            agent_inputs (torch.Tensor): Batch of agent observations.
                                         Shape: (batch_size * n_agents, input_shape).
            h_in (torch.Tensor): Batch of current GRU hidden states.
                                 Shape: (batch_size * n_agents, rnn_hidden_dim).

        Returns:
            torch.Tensor: New hidden state from the GRU cell,
                                       shape (batch_size * n_agents, rnn_hidden_dim).
        """
        x = F.relu(self.fc1(agent_inputs)) # Shape: (batch*n_agents, rnn_hidden_dim)
        
        # Ensure tensors are contiguous before GRUCell call
        x = x.contiguous()
        h_in = h_in.contiguous()
        if h_in.device != x.device:
             h_in = h_in.to(x.device)
        # Check shape consistency if needed (removed for brevity, assume MAC handles it)
        # if h_in.shape[0] != x.shape[0] or h_in.shape[1] != x.shape[1]:
        #      raise ValueError(f"GRUCell shape mismatch: x={x.shape}, h_in={h_in.shape}")

        h = self.rnn(x, h_in) # Shape: (batch*n_agents, rnn_hidden_dim)
        return h

    def actor_forward(self, inputs):
        """
        Forward pass for the Actor network component.

        Args:
            inputs (torch.Tensor): Batch of agent observations, shape (batch_size, input_shape).
                                     Note: batch_size here often refers to (batch_size * n_agents).

        Returns:
            torch.Tensor: Continuous parameters (e.g., normalized power levels) 
                          for all discrete actions, shape (batch_size, n_actions).
        """
        continuous_params = self.actor(inputs) # Shape: (batch_size, n_actions)
        return continuous_params

    def get_q_value_for_action(self, hidden_state, discrete_action_index, continuous_param):
        """
        Calculates the Q-value for a specific discrete action and continuous parameter,
        given the current hidden state.

        Args:
            hidden_state (torch.Tensor): The hidden state obtained from forward(),
                                         shape (batch_size, rnn_hidden_dim).
            discrete_action_index (torch.Tensor): Index of the discrete action chosen,
                                                  shape (batch_size, 1).
            continuous_param (torch.Tensor): Continuous parameter corresponding to the chosen action,
                                              shape (batch_size, 1).

        Returns:
            torch.Tensor: The calculated Q-value for the given input, shape (batch_size, 1).
        """
        batch_size = hidden_state.shape[0]

        # One-hot encode the discrete action index
        # Ensure index is long type for one_hot
        discrete_action_index = discrete_action_index.long()
        # Handle potential extra dimensions if index is (batch, 1) instead of (batch,)
        if discrete_action_index.dim() > 1 and discrete_action_index.shape[1] == 1:
            discrete_action_index = discrete_action_index.squeeze(1)
        
        # Check if indices are within bounds before one-hot encoding
        if torch.any(discrete_action_index < 0) or torch.any(discrete_action_index >= self.n_actions):
             raise IndexError(f"Action index out of bounds: {discrete_action_index}, n_actions: {self.n_actions}")

        action_one_hot = F.one_hot(discrete_action_index, num_classes=self.n_actions).float()
        # Ensure shape is (batch_size, n_actions)
        action_one_hot = action_one_hot.view(batch_size, self.n_actions)

        # Ensure continuous_param has shape (batch_size, 1)
        if continuous_param.dim() == 1:
             continuous_param = continuous_param.unsqueeze(1)
        elif continuous_param.dim() > 2 or (continuous_param.dim() == 2 and continuous_param.shape[1] != 1):
             # Attempt to reshape if possible, otherwise raise error
             try:
                 continuous_param = continuous_param.view(batch_size, 1)
             except RuntimeError:
                  raise ValueError(f"Unexpected continuous_param shape: {continuous_param.shape}, expected ({batch_size}, 1)")

        # Concatenate hidden state, one-hot action, and continuous parameter
        q_head_input = torch.cat([hidden_state, action_one_hot, continuous_param], dim=1)

        # Pass through the Q-value head
        q_value = self.fc2_q_head(q_head_input) # Shape: (batch_size, 1)

        return q_value

class QMixer(nn.Module):
    """
    QMix Mixing Network.

    Combines individual agent Q-values (chosen based on their actions) into a 
    joint action-value Q_tot using a monotonic mixing function.
    The weights and biases of the mixing network are generated dynamically 
    based on the global state using hypernetworks.
    Ensures that argmax Q_tot = argmax Q_i for each agent (Individual-Global Max property).
    
    Reference: QMix: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning (Rashid et al., 2018)
    https://arxiv.org/abs/1803.11485
    """
    def __init__(self, args):
        """
        Initializes the QMixer network.

        Args:
            args (SimpleNamespace): Configuration object containing necessary parameters
                                     (n_agents, state_shape, mixing_embed_dim, hyper_hidden_dim).
        """
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        # Calculate the flattened size of the global state vector
        self.state_dim = int(np.prod(args.state_shape)) 

        # Dimensionality parameters for the mixing network and hypernetworks
        self.embed_dim = args.mixing_embed_dim # Output dim of the first mixing layer
        self.hyper_hidden_dim = args.hyper_hidden_dim # Hidden dim of the hypernetworks

        # --- Add LayerNorm for state normalization ---
        self.state_norm = nn.LayerNorm(self.state_dim)

        # --- Hypernetworks --- 
        # These networks take the global state `s` and output the weights/biases 
        # for the main mixing network layers.
        
        # Hypernetwork for the weights of the first mixing layer (W1)
        # Input: state_dim, Output: n_agents * embed_dim
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.n_agents * self.embed_dim)
        )
        
        # Hypernetwork for the weights of the second (final) mixing layer (W2)
        # Input: state_dim, Output: embed_dim * 1 (since output Q_tot is scalar)
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.embed_dim * 1) # Reshaped later
        )

        # Hypernetwork for the bias of the first mixing layer (b1)
        # Input: state_dim, Output: embed_dim
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        
        # Hypernetwork for the state-dependent bias of the final layer V(s)
        # This acts like the bias term for the second layer (b2).
        # Input: state_dim, Output: 1 (scalar bias for Q_tot)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim), # Hidden layer in the bias hypernet
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        Forward pass of the QMixer.

        Args:
            agent_qs (torch.Tensor): Q-values for the actions taken by each agent.
                                     Shape: (batch_size * seq_len, n_agents) or (batch_size, n_agents).
                                     **Important: These should be the Q(s, T_chosen, P_chosen) values.**
            states (torch.Tensor): Global state. 
                                   Shape: (batch_size * seq_len, state_dim) or (batch_size, state_dim).

        Returns:
            torch.Tensor: Computed global Q_tot value. 
                          Shape: (batch_size * seq_len, 1) or (batch_size, 1).
        """
        # Get batch size (might include sequence length if processing batches of transitions)
        batch_size = agent_qs.size(0)
        # Reshape states to be flat (batch_size * seq_len, state_dim)
        states = states.reshape(-1, self.state_dim)
        # --- Normalize the state ---
        states_normalized = self.state_norm(states)

        # Reshape agent Q-values for batch matrix multiplication 
        # Shape: (batch_size * seq_len, 1, n_agents)
        agent_qs = agent_qs.view(-1, 1, self.n_agents) 

        # --- Generate Mixing Network Weights and Biases using NORMALIZED state ---
        # Generate weights for the first layer (W1) using hyper_w_1
        # w1 = torch.abs(self.hyper_w_1(states)) # Original: Enforce non-negativity constraint (monotonicity)
        # Clamp weights to a reasonable range [0, 1] to prevent explosion while maintaining monotonicity
        w1 = torch.clamp(self.hyper_w_1(states_normalized), min=0.0, max=5.0) # Clamp W1
        # Reshape W1 to (batch_size * seq_len, n_agents, embed_dim)
        w1 = w1.view(-1, self.n_agents, self.embed_dim) 

        # Generate bias for the first layer (b1) using hyper_b_1
        b1 = self.hyper_b_1(states_normalized)
        b1 = torch.clamp(b1, min=-5.0, max=5.0) # Clamp b1 to prevent explosion
        # Reshape b1 to (batch_size * seq_len, 1, embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim) 

        # Generate weights for the final layer (W_final) using hyper_w_final
        # w_final = torch.abs(self.hyper_w_final(states)) # Original: Enforce non-negativity constraint
        w_final = torch.clamp(self.hyper_w_final(states_normalized), min=0.0, max=5.0) # Clamp W_final
        # Reshape W_final to (batch_size * seq_len, embed_dim, 1)
        w_final = w_final.view(-1, self.embed_dim, 1) 

        # Generate state-dependent bias V(s) using V
        v = self.V(states_normalized)
        v = torch.clamp(v, min=-5.0, max=5.0) # Clamp V(s)
        v = v.view(-1, 1, 1) # Shape: (batch_size * seq_len, 1, 1)

        # --- Apply Mixing Network Layers --- 
        # First layer: Q_i * W1 + b1
        # Apply ELU activation (or ReLU as per some implementations)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1) # Shape: (batch_size * seq_len, 1, embed_dim)
        
        # Second layer: hidden * W_final + V(s)
        y = torch.bmm(hidden, w_final) + v # Shape: (batch_size * seq_len, 1, 1)

        # Reshape final output to (batch_size * seq_len, 1)
        q_tot = y.view(batch_size, -1, 1) 
        
        # If the input was just a single time step (seq_len=1), remove the middle dimension
        if q_tot.shape[1] == 1:
            q_tot = q_tot.squeeze(1) # Shape: (batch_size, 1)
            
        return q_tot 