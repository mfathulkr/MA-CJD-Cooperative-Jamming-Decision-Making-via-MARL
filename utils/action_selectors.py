import torch
import numpy as np

class EpsilonGreedyActionSelector:
    """
    Implements epsilon-greedy action selection.
    """
    def __init__(self, args):
        self.args = args
        self.epsilon_start = args.epsilon_start
        self.epsilon_finish = args.epsilon_finish
        self.epsilon_anneal_time = args.epsilon_anneal_time
        self.epsilon = self.epsilon_start

    def select_action(self, agent_qs, avail_actions, t_env, test_mode=False):
        """
        Selects discrete actions based on Q-values and exploration strategy.

        Args:
            agent_qs (torch.Tensor): Q-values for each agent (batch_size, n_agents, n_actions).
            avail_actions (torch.Tensor): Mask of available actions (batch_size, n_agents, n_actions).
            t_env (int): Current environment time step for epsilon annealing.
            test_mode (bool): If True, disable exploration (greedy selection).

        Returns:
            torch.Tensor: Chosen discrete actions (batch_size, n_agents, 1).
        """
        
        # Anneal epsilon
        if not test_mode:
            delta = (self.epsilon_start - self.epsilon_finish) / self.epsilon_anneal_time
            self.epsilon = max(self.epsilon_finish, self.epsilon_start - delta * t_env)

        # Mask unavailable actions
        masked_qs = agent_qs.clone()
        masked_qs[avail_actions == 0] = -float("inf") # Set Q-value of unavailable actions to -inf

        # Epsilon-greedy selection
        random_numbers = torch.rand_like(agent_qs[:, :, 0]) # Random numbers for each agent in the batch
        pick_random = (random_numbers < self.epsilon).long()
        
        # Get random actions where needed (consider only available actions)
        # Need to handle selection of random actions from available ones carefully
        # A simpler approach is to pick randomly from *all* actions and rely on masking later if necessary,
        # or ensure random choice respects availability.
        # Let's use torch.multinomial on available actions for random choice.
        avail_actions_float = avail_actions.float()
        # Ensure valid probabilities for multinomial (at least one action available)
        avail_actions_float[(avail_actions_float.sum(dim=-1) == 0)] = 1.0 / avail_actions_float.shape[-1] # Uniform if no actions available (should not happen in valid env)
        
        random_actions = torch.multinomial(avail_actions_float.view(-1, agent_qs.shape[-1]), num_samples=1).view(agent_qs.shape[0], agent_qs.shape[1])

        # Get greedy actions
        greedy_actions = masked_qs.argmax(dim=2)

        # Combine random and greedy actions
        chosen_actions = pick_random * random_actions + (1 - pick_random) * greedy_actions
        
        if test_mode:
            # Always choose greedy action in test mode
            chosen_actions = masked_qs.argmax(dim=2)

        return chosen_actions.unsqueeze(-1) # Add dimension for consistency 