import os
from typing import Iterable

import cv2
import numpy as np
import plotly
import torch
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
from torch.nn import Module
from torch.nn import functional as F


# _epsilon = 1e-1
# time step of the gym environment 


# Plots min, max and mean + standard deviation bars of a population over time
def lineplot(xs, ys_population, title, path='', xaxis='episode'):
    max_colour, mean_colour, std_colour, transparent = (
        'rgb(0, 132, 180)',
        'rgb(0, 172, 237)',
        'rgba(29, 202, 255, 0.2)',
        'rgba(0, 0, 0, 0)',
    )

    if isinstance(ys_population[0], list) or isinstance(ys_population[0], tuple):
        ys = np.asarray(ys_population, dtype=np.float32)
        ys_min, ys_max, ys_mean, ys_std, ys_median = ys.min(1), ys.max(1), ys.mean(1), ys.std(1), np.median(ys, 1)
        ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

        trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
        trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
        trace_mean = Scatter(
            x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean'
        )
        trace_lower = Scatter(
            x=xs,
            y=ys_lower,
            fill='tonexty',
            fillcolor=std_colour,
            line=Line(color=transparent),
            name='-1 Std. Dev.',
            showlegend=False,
        )
        trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')
        trace_median = Scatter(x=xs, y=ys_median, line=Line(color=max_colour), name='Median')
        data = [trace_upper, trace_mean, trace_lower, trace_min, trace_max, trace_median]
    else:
        data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour))]
    plotly.offline.plot(
        {'data': data, 'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})},
        filename=os.path.join(path, title + '.html'),
        auto_open=False,
    )


def write_video(frames, title, path=''):
    frames = (
        np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :, ::-1]
    )  # VideoWrite expects H x W x C in BGR
    _, H, W, _ = frames.shape
    writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H), True)
    for frame in frames:
        writer.write(frame)
    writer.release()


def imagine_ahead(prev_state, prev_belief, policy, transition_model, planning_horizon=12):
    '''
    imagine_ahead is the function to draw the imaginary tracjectory using the dynamics model and policy.
    Input: current state (posterior), current belief (hidden), policy, transition_model  
    Output: generated trajectory of features includes beliefs, prior_states, prior_means, prior_std_devs
            torch.Size([2450, 200]) torch.Size([2450, 30]) torch.Size([2450, 30]) torch.Size([2450, 30])
    '''
    flatten = lambda x: x.view([-1] + list(x.size()[2:]))
    prev_belief = flatten(prev_belief)
    prev_state = flatten(prev_state)
    # print(prev_state.size())

    # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    T = planning_horizon
    beliefs, prior_states, prior_means, prior_std_devs = (
        [torch.empty(0)] * T,
        [torch.empty(0)] * T,
        [torch.empty(0)] * T,
        [torch.empty(0)] * T,
    )
    beliefs[0], prior_states[0] = prev_belief, prev_state
    # print(len(prev_belief))

    # Loop over time sequence
    for t in range(T - 1):
        _state = prior_states[t]
        actions = policy.get_action(_state.detach())
        # Compute belief (deterministic hidden state)
        hidden = transition_model.act_fn(transition_model.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
        beliefs[t + 1] = transition_model.rnn(hidden, beliefs[t])
        # Compute state prior by applying transition dynamics
        hidden = transition_model.act_fn(transition_model.fc_embed_belief_prior(beliefs[t + 1]))
        prior_means[t + 1], _prior_std_dev = torch.chunk(transition_model.fc_state_prior(hidden), 2, dim=1)
        prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + transition_model.min_std_dev
        prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
        # print(len(beliefs[t]))
    # Return new hidden states
    # imagined_traj = [beliefs, prior_states, prior_means, prior_std_devs]
    imagined_traj = [
        torch.stack(beliefs[1:], dim=0),
        torch.stack(prior_states[1:], dim=0),
        torch.stack(prior_means[1:], dim=0),
        torch.stack(prior_std_devs[1:], dim=0),
    ]
    return imagined_traj


def lambda_return(imged_reward, value_pred, bootstrap, discount=0.99, lambda_=0.95):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
    discount_tensor = discount * torch.ones_like(imged_reward)  # pcont
    inputs = imged_reward + discount_tensor * next_values * (1 - lambda_)
    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []
    for index in indices:
        inp, disc = inputs[index], discount_tensor[index]
        last = inp + disc * lambda_ * last
        outputs.append(last)
    outputs = list(reversed(outputs))
    outputs = torch.stack(outputs, 0)
    returns = outputs
    # print(returns)
    return returns

# Out-dated Barrier Loss Calculation Function 
def barrier_loss_stoch_return(imged_cost, barrier_pred, COST_THRESHOLD,  _omega):
    state_filter_mask = COST_THRESHOLD * torch.ones_like(imged_cost)
    # Safe state should be 0 in the unsafe_mask, and Unsafe state should be 1
    unsafe_mask = F.relu(imged_cost - state_filter_mask)
    cost_barrier_unsafe_mask = unsafe_mask.bool().int()
    # Unsafe state should be 0 in the safe_mask, and Unsafe state should be 1
    safe_mask = F.relu(imged_cost - state_filter_mask)
    cost_barrier_safe_mask = safe_mask.bool().int()
    # Enforce supermartingale in increasing expectation
    barrier_after = barrier_pred[1:]
    diff_expactation = []
    for i in range(len(barrier_after)):
        diff_expactation.append(barrier_after[i] - barrier_pred[i])
    diff_expactation = torch.stack(diff_expactation, 0)
    supp = torch.zeros((1, barrier_pred.shape[1]), device='cuda')
    cost_barrier_expectation = torch.cat([supp, diff_expactation])
    # Enforce the unsafe barrier function value should be larger than 1
    unsafe_cost = cost_barrier_unsafe_mask - barrier_pred * cost_barrier_unsafe_mask
    # Enforce the safe barrier function value should be smaller than eta
    safe_cost = barrier_pred * cost_barrier_safe_mask - _omega * cost_barrier_safe_mask 
    # Sum up all the loss
    losses = cost_barrier_expectation + unsafe_cost + safe_cost
    return losses


# Tensor Calculation based faster version barrier loss function
def barrier_loss_return(imged_cost, barrier_prev, COST_THRESHOLD, _epsilon, _DT = 0.02):
    # imged_cost = torch.transpose(imged_cost, 0, 1)
    # barrier_prev = torch.transpose(barrier_prev, 0, 1)
    state_filter_mask = COST_THRESHOLD * torch.ones_like(imged_cost)
    sigma = 0.0001 * torch.ones_like(barrier_prev)
    # Safe state should be 0 in the unsafe_mask
    unsafe_mask = F.relu(imged_cost - state_filter_mask)
    # Unsafe state should be 0 in the safe_mask
    safe_mask = F.relu(state_filter_mask - imged_cost)
    barrier_after = barrier_prev[1:]
    derivative = []
    for i in range(len(barrier_after)):
        derivative.append((barrier_after[i] - barrier_prev[i])/_DT)
    derivative = torch.stack(derivative, 0)
    cost_barrier = _epsilon * torch.ones_like(barrier_prev)
    # Negative derivative satisfy conditions, only penalize the positive case
    cost_barrier_derivative = F.relu(cost_barrier[1:] - derivative - barrier_prev[1:])
    supp = torch.zeros((1, cost_barrier_derivative.shape[1]), device='cuda')
    cost_barrier_derivative = torch.cat([supp, cost_barrier_derivative])
    # print(cost_barrier_derivative.grad_fn)
    # Safe state is 0, correctly categorized unsafe state is negative, and wrongly categorized unsafe state is positive
    cost_barrier_unsafe_mask = torch.div(F.relu(barrier_prev * unsafe_mask), barrier_prev+sigma)
    cost_barrier_unsafe_mask = cost_barrier_unsafe_mask.bool().int()
    # print(torch.all(cost_barrier_unsafe_mask>=0))
    epsilon_unsafe_mask = _epsilon * cost_barrier_unsafe_mask
    cost_barrier_unsafe = cost_barrier_unsafe_mask * barrier_prev + epsilon_unsafe_mask
    # print(torch.all(cost_barrier_unsafe>=0))
    # print(cost_barrier_unsafe.grad_fn)
    # Unsafe state is 0, correctly categorized safe state is positive, and wrongly categorized safe state is negative
    cost_barrier_safe_mask = torch.div(F.relu(- safe_mask * barrier_prev), barrier_prev+sigma)
    cost_barrier_safe_mask = cost_barrier_safe_mask.bool().int()
    epsilon_safe_mask = _epsilon * cost_barrier_safe_mask
    cost_barrier_safe = epsilon_safe_mask - cost_barrier_safe_mask * barrier_prev
    # print(cost_barrier_safe.grad_fn)
    # Ensemble all the cost together
    loss = 1e2 * cost_barrier_derivative + cost_barrier_safe + cost_barrier_unsafe
    # print(f'safe_cost: {torch.mean(torch.sum(cost_barrier_safe, dim=0))}\n')
    # print(f'unsafe_cost: {torch.mean(torch.sum(cost_barrier_unsafe, dim=0))}\n')
    # print(f'barrier_deriv: {torch.mean(torch.sum(cost_barrier_derivative, dim=0))}\n')
    # print(loss)
    return loss


class ActivateParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally Activate the gradients.
        example:
        ```
        with ActivateParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            # print(param.requires_grad)
            param.requires_grad = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


# "get_parameters" and "FreezeParameters" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]
