
import numpy as np
import torch
import torch.distributions
from torch import jit, nn
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.nn import functional as F


# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner(jit.ScriptModule):
    __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates']

    def __init__(
        self,
        action_size,
        planning_horizon,
        optimisation_iters,
        candidates,
        top_candidates,
        transition_model,
        reward_model,
    ):
        super().__init__()
        self.transition_model, self.reward_model = transition_model, reward_model
        self.action_size = action_size
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates

    @jit.script_method
    def forward(self, belief, state):
        B, H, Z = belief.size(0), belief.size(1), state.size(1)
        belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(
            dim=1
        ).expand(B, self.candidates, Z).reshape(-1, Z)
        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        action_mean, action_std_dev = torch.zeros(
            self.planning_horizon, B, 1, self.action_size, device=belief.device
        ), torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)
        for _ in range(self.optimisation_iters):
            # print("optimization_iters",_)
            # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
            actions = (
                action_mean
                + action_std_dev
                * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=action_mean.device)
            ).view(
                self.planning_horizon, B * self.candidates, self.action_size
            )  # Sample actions (time x (batch x candidates) x actions)
            # Sample next states
            beliefs, states, _, _ = self.transition_model(
                state, actions, belief
            )  # [12, 1000, 200] [12, 1000, 30] : 12 horizon steps; 1000 candidates
            # Calculate expected returns (technically sum of rewards over planning horizon)
            returns = (
                self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).sum(dim=0)
            )  # output from r-model[12000]->view[12, 1000]->sum[1000]
            # Re-fit belief to the K best action sequences
            _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
            topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(
                dim=1
            )  # Fix indices for unrolled actions
            best_actions = actions[:, topk.view(-1)].reshape(
                self.planning_horizon, B, self.top_candidates, self.action_size
            )
            # Update belief with new means and standard deviations
            action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(
                dim=2, unbiased=False, keepdim=True
            )
        # Return first action mean µ_t
        return action_mean[0].squeeze(dim=1)


class Controller(jit.ScriptModule):
    def __init__(
        self,
        # belief_size,
        state_size,
        hidden_size,
        action_size,
        activation_function='elu',
    ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, action_size)
        self.modules = [self.fc1, self.fc2, self.fc5]

    @jit.script_method
    def forward(self, state):
        x = state
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        # hidden = self.act_fn(self.fc3(hidden))
        # hidden = self.act_fn(self.fc4(hidden))
        action = self.fc5(hidden).squeeze(dim=1)
        return action 

    def get_action(self, state):
        return self.forward(state)


class BarrierNN(jit.ScriptModule):
    def __init__(self, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

    @jit.script_method
    def forward(self, state):
        x = state
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        barrier = self.fc4(hidden).squeeze(dim=1)
        return barrier
    

class BarrierNN_stoch(jit.ScriptModule):

    def __init__(self, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

    @jit.script_method
    def forward(self, state):
        x = state
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        barrier = nn.Sigmoid(self.fc4(hidden)).squeeze(dim=1)
        return barrier
    

class TargetSet_Func(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        # self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, state):
        x = state
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        # hidden = self.act_fn(self.fc3(hidden))
        val = self.fc3(hidden).squeeze(dim=1)
        return val

class AvoidSet_Func(jit.ScriptModule):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_sizee, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        # self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3]

    @jit.script_method
    def forward(self, state):
        x = state
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        # hidden = self.act_fn(self.fc3(hidden))
        val = self.fc3(hidden).squeeze(dim=1)
        return val 