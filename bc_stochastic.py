import argparse
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from modules.env import CONTROL_SUITE_ENVS, GYM_ENVS, SAFETY_GYM_ENVS, Env, EnvBatcher
from utils.memory import ExperienceReplay
from modules.models import Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, bottle, CostModel, Controller_stoch, get_through_NN
from modules.planner import MPCPlanner, Controller, BarrierNN
from utils.utils import FreezeParameters, lambda_return, lineplot, write_video, imagine_ahead, barrier_loss_stoch_return

# _eta = 0.01

# Hyperparameters
parser = argparse.ArgumentParser(description='CBF-Dreamer')
parser.add_argument('--algo', type=str, default='cbf-dreamer', help='cbf-dreamer')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')

parser.add_argument('--cost_threshold', type=float, default=0.01, help='Threshold to distinguish safe and unsafe region')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument(
    '--env',
    type=str,
    default='Pendulum-v0',
    choices=SAFETY_GYM_ENVS,
    help='Safety_GYM_Env',
)
parser.add_argument('--eta', type=float, default=1, help='Eta on safety parameter')
parser.add_argument('--epsilon', type=float, default=1e-2, help='Margin used to find bc')
parser.add_argument('--observation_type', default='rgb_image')
parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument(
    '--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size'
)  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument(
    '--cnn-activation-function',
    type=str,
    default='relu',
    choices=dir(F),
    help='Model activation function for a convolution layer',
)
parser.add_argument(
    '--dense-activation-function',
    type=str,
    default='elu',
    choices=dir(F),
    help='Model activation function a dense layer',
)
parser.add_argument(
    '--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size'
)  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.1, metavar='ε', help='Action noise')

parser.add_argument('--episodes', type=int, default=500, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=1, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=1000, metavar='C', help='Training steps of each episode')

parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=32, metavar='L', help='Chunk size')
parser.add_argument(
    '--worldmodel-LogProbLoss',
    action='store_true',
    help='use LogProb loss for observation_model and reward_model training',
)
parser.add_argument(
    '--overshooting-distance',
    type=int,
    default=50,
    metavar='D',
    help='Latent overshooting distance/latent overshooting weight for t = 1',
)
parser.add_argument(
    '--overshooting-kl-beta',
    type=float,
    default=0,
    metavar='β>1',
    help='Latent overshooting KL weight for t > 1 (0 to disable)',
)
parser.add_argument(
    '--overshooting-reward-scale',
    type=float,
    default=0,
    metavar='R>1',
    help='Latent overshooting reward prediction weight for t > 1 (0 to disable)',
)
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--model_learning_rate', type=float, default=1e-4, metavar='α', help='Learning rate')

parser.add_argument('--value_learning_rate', type=float, default=4e-4, metavar='α', help='Learning rate')
parser.add_argument('--barrier_learning_rate', type=float, default=8e-4, metavar='α', help='Learning rate')
parser.add_argument('--controller_learning_rate', type=float, default=1e-3, metavar='α', help='Learning rate')
parser.add_argument('--cbf_learning_rate', type=float, default=4e-4, metavar='α', help='Learning rate')
parser.add_argument(
    '--learning-rate-schedule',
    type=int,
    default=0,
    metavar='αS',
    help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)',
)
parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value')
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')

parser.add_argument('--planning-horizon', type=int, default=30, metavar='H', help='Planning horizon distance')
parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')
parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
parser.add_argument('--test', action='store_true', help='Test only')

parser.add_argument('--test-interval', type=int, default=10, metavar='I', help='Test interval (episodes)')
parser.add_argument('--test-episodes', type=int, default=50, metavar='E', help='Number of test episodes')
parser.add_argument('--video-interval', type=int, default=100, metavar='VI', help='Interval to write video')

parser.add_argument('--checkpoint-interval', type=int, default=200, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
args = parser.parse_args()
args.overshooting_distance = min(
    args.chunk_size, args.overshooting_distance
)  # Overshooting distance cannot be greater than chunk size
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))


# Setup
results_dir = os.path.join('results', '{}_{}'.format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
    print("using CUDA")
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
else:
    print("using CPU")
    args.device = torch.device('cpu')
metrics = {
    'steps': [],
    'episodes': [],
    'train_rewards': [],
    'train_costs': [],
    'test_episodes': [],
    'test_rewards': [],
    'test_costs': [],
    'observation_loss': [],
    'reward_loss': [],
    'cost_loss': [],
    'kl_loss': [],
    'barrier_loss': [],
    'controller_loss': [],
    'value_loss': [],
    'average_violation': [],
}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.id))
print("writer is ready")

# Initialise training environment and experience replay memory
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
print("environment is loaded")
if args.experience_replay != '' and os.path.exists(args.experience_replay):
    D = torch.load(args.experience_replay)
    metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
    print("experience replay buffer is ready")
elif not args.test:
    D = ExperienceReplay(
        args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.bit_depth, args.device
    )
    # Initialise dataset D with S random seed episodes
    for s in range(1, args.seed_episodes + 1):
        observation, done, t = env.reset(), False, 0
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done, cost = env.step(action)
            D.append(observation, action, reward, done, cost)
            observation = next_observation
            t += 1
        metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
        metrics['episodes'].append(s)
    print("experience replay buffer is ready")


# Initialise model parameters randomly
transition_model = TransitionModel(
    args.belief_size,
    args.state_size,
    env.action_size,
    args.hidden_size,
    args.embedding_size,
    args.dense_activation_function,
).to(device=args.device)

observation_model = ObservationModel(
    args.symbolic_env,
    env.observation_size,
    args.belief_size,
    args.state_size,
    args.embedding_size,
    args.cnn_activation_function,
).to(device=args.device)

reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(
    device=args.device
)

cost_model = CostModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(
    device=args.device
)

barrier_model = BarrierNN(args.state_size, args.hidden_size).to(device=args.device)

encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.cnn_activation_function).to(
    device=args.device
)

# controller = Controller(args.belief_size, args.state_size, args.hidden_size, env.action_size).to(device=args.device)

controller = Controller_stoch(
    args.state_size, args.hidden_size, env.action_size, args.dense_activation_function
).to(device=args.device)

value_model = ValueModel(args.state_size, args.hidden_size, args.dense_activation_function).to(
    device=args.device
)

param_list = (
    list(transition_model.parameters())
    + list(observation_model.parameters())
    + list(reward_model.parameters())
    + list(cost_model.parameters())
    + list(encoder.parameters())
)
value_barrier_controller_param_list = list(value_model.parameters()) + list(barrier_model.parameters()) + list(controller.parameters())

cbf_params_list = (
    list(barrier_model.parameters()) 
    + list(controller.parameters())
)
params_list = param_list + value_barrier_controller_param_list
print("transition, observation, reward, encoder, barrier, controller, value models are ready")

model_optimizer = optim.Adam(
    param_list, lr=0 if args.learning_rate_schedule != 0 else args.model_learning_rate, eps=args.adam_epsilon
)

barrier_optimizer = optim.Adam(
    barrier_model.parameters(),
    lr=0 if args.learning_rate_schedule != 0 else args.barrier_learning_rate,
    eps=args.adam_epsilon,
)

controller_optimizer = optim.Adam(
    barrier_model.parameters(),
    lr=0 if args.learning_rate_schedule != 0 else args.controller_learning_rate,
    eps=args.adam_epsilon,
)

cbf_optimizer = optim.Adam(
    cbf_params_list,
    lr=0 if args.learning_rate_schedule != 0 else args.cbf_learning_rate,
    eps=args.adam_epsilon,
)

value_optimizer = optim.Adam(
    value_model.parameters(),
    lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate,
    eps=args.adam_epsilon,
)

if args.models != '' and os.path.exists(args.models):
    print("loading pre-trained models")
    model_dicts = torch.load(args.models)
    transition_model.load_state_dict(model_dicts['transition_model'])
    observation_model.load_state_dict(model_dicts['observation_model'])
    reward_model.load_state_dict(model_dicts['reward_model'])
    cost_model.load_state_dict(model_dicts['cost_model'])
    encoder.load_state_dict(model_dicts['encoder'])
    # barrier_model.load_state_dict(model_dicts['barrier_model'])
    controller.load_state_dict(model_dicts['controller'])
    value_model.load_state_dict(model_dicts['value_model'])
    model_optimizer.load_state_dict(model_dicts['model_optimizer'])


print("CBF-Dreamer")

global_prior = Normal(
    torch.zeros(args.batch_size, args.state_size, device=args.device),
    torch.ones(args.batch_size, args.state_size, device=args.device),
)  # Global prior N(0, I)
free_nats = torch.full((1,), args.free_nats, device=args.device)  # Allowed deviation in KL divergence
print("models and planners are ready")


def update_belief_and_act(
    args, env, planner, transition_model, encoder, belief, posterior_state, action, observation, explore=False
):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    # print("action size: ",action.size()) torch.Size([1, 6])
    belief, _, _, _, posterior_state, _, _ = transition_model(
        posterior_state, action.unsqueeze(dim=0), belief, encoder(observation).unsqueeze(dim=0)
    )  # Action and observation need extra time dimension
    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(
        dim=0
    )  # Remove time dimension from belief/state
   
    # # Input state for planner forward and generate action
    action = planner.get_action(posterior_state, det=explore)

    if explore:
        action = torch.clamp(
            Normal(action, args.action_noise).rsample(), -1, 1
        )  # Add gaussian exploration noise on top of the sampled action
        # action = action + args.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
    next_observation, reward, done, cost = env.step(
        action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu()
    )  # Perform environment step (action repeats handled internally)
    return belief, posterior_state, action, next_observation, reward, done, cost


# Testing only
if args.test:
    # Set models to eval mode
    transition_model.eval()
    observation_model.eval()
    reward_model.eval()
    cost_model.eval()
    encoder.eval()
    controller.eval()
    value_model.eval()
    with torch.no_grad():
        total_reward = 0
        total_cost = 0
        total_violation = 0
        for _ in tqdm(range(args.test_episodes)):
            observation = env.reset()
            belief, posterior_state, action = (
                torch.zeros(1, args.belief_size, device=args.device),
                torch.zeros(1, args.state_size, device=args.device),
                torch.zeros(1, env.action_size, device=args.device),
            )
            pbar = tqdm(range(args.max_episode_length // args.action_repeat))
            for t in pbar:
                belief, posterior_state, action, observation, reward, done, cost = update_belief_and_act(
                    args,
                    env,
                    controller,
                    transition_model,
                    encoder,
                    belief,
                    posterior_state,
                    action,
                    observation.to(device=args.device),
                    explore=False
                )
                if cost > args.cost_threshold:
                    total_violation += 1
                total_reward += reward
                total_cost += cost
                if args.render:
                    env.render()
                if done:
                    pbar.close()
                    break
    print('Average Reward:', total_reward / args.test_episodes)
    print('Average Cost:', total_cost / args.test_episodes)
    print('Average Violation:', total_violation / args.test_episodes)
    env.close()
    quit()


# Training (and testing)
for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics['episodes'][-1] + 1):
    # Model fitting
    losses = []
    model_modules = transition_model.modules + encoder.modules + observation_model.modules + reward_model.modules + cost_model.modules
    cbf_modules = controller.modules + barrier_model.modules
    print("training loop")
    # temp = torch.zeros(1, device=args.device)
    for s in tqdm(range(args.collect_interval)):
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, costs, nonterminals = D.sample(
            args.batch_size, args.chunk_size
        )  # Transitions start at time t = 0
        # Create initial belief and state for time t = 0
        init_belief, init_state = torch.zeros(args.batch_size, args.belief_size, device=args.device), torch.zeros(
            args.batch_size, args.state_size, device=args.device
        )
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        (
            beliefs,
            prior_states,
            prior_means,
            prior_std_devs,
            posterior_states,
            posterior_means,
            posterior_std_devs,
        ) = transition_model(
            init_state, actions[:-1], init_belief, bottle(encoder, (observations[1:],)), nonterminals[:-1]
        )
        # print(posterior_states.size())
        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        # Observation loss
        if args.worldmodel_LogProbLoss:
            observation_dist = Normal(bottle(observation_model, (beliefs, posterior_states)), 1)
            observation_loss = (
                -observation_dist.log_prob(observations[1:])
                .sum(dim=2 if args.symbolic_env else (2, 3, 4))
                .mean(dim=(0, 1))
            )
        else:
            observation_loss = (
                F.mse_loss(bottle(observation_model, (beliefs, posterior_states)), observations[1:], reduction='none')
                .sum(dim=2 if args.symbolic_env else (2, 3, 4))
                .mean(dim=(0, 1))
            )

        # Reward loss
        if args.worldmodel_LogProbLoss:
            reward_dist = Normal(bottle(reward_model, (beliefs, posterior_states)), 1)
            reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
        else:
            reward_loss = F.mse_loss(
                bottle(reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none'
            ).mean(dim=(0, 1))

        # Cost Loss
        if args.worldmodel_LogProbLoss:
            cost_dist = Normal(bottle(cost_model, (beliefs, posterior_states)), 1)
            cost_loss = -cost_dist.log_prob(costs[:-1]).mean(dim=(0, 1))
        else:
            cost_loss = F.mse_loss(
                bottle(cost_model, (beliefs, posterior_states)), costs[:-1], reduction='none'
            ).mean(dim=(0, 1))

        # Transition Loss
        div = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2)
        kl_loss = torch.max(div, free_nats).mean(
            dim=(0, 1)
        )  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
        if args.global_kl_beta != 0:
            kl_loss += args.global_kl_beta * kl_divergence(
                Normal(posterior_means, posterior_std_devs), global_prior
            ).sum(dim=2).mean(dim=(0, 1))

        # Calculate latent overshooting objective for t > 0
        if args.overshooting_kl_beta != 0:
            overshooting_vars = []  # Collect variables for overshooting to process in batch
            for t in range(1, args.chunk_size - 1):
                d = min(t + args.overshooting_distance, args.chunk_size - 1)  # Overshooting distance
                t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
                seq_pad = (
                    0,
                    0,
                    0,
                    0,
                    0,
                    t - d + args.overshooting_distance,
                )  # Calculate sequence padding so overshooting terms can be calculated in one batch
                # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
                overshooting_vars.append(
                    (
                        F.pad(actions[t:d], seq_pad),
                        F.pad(nonterminals[t:d], seq_pad),
                        F.pad(rewards[t:d], seq_pad[2:]),
                        beliefs[t_],
                        prior_states[t_],
                        F.pad(posterior_means[t_ + 1 : d_ + 1].detach(), seq_pad),
                        F.pad(posterior_std_devs[t_ + 1 : d_ + 1].detach(), seq_pad, value=1),
                        F.pad(torch.ones(d - t, args.batch_size, args.state_size, device=args.device), seq_pad),
                    )
                )  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
            overshooting_vars = tuple(zip(*overshooting_vars))
            # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
            beliefs, prior_states, prior_means, prior_std_devs = transition_model(
                torch.cat(overshooting_vars[4], dim=0),
                torch.cat(overshooting_vars[0], dim=1),
                torch.cat(overshooting_vars[3], dim=0),
                None,
                torch.cat(overshooting_vars[1], dim=1),
            )
            seq_mask = torch.cat(overshooting_vars[7], dim=1)
            # Calculate overshooting KL loss with sequence mask
            kl_loss += (
                (1 / args.overshooting_distance)
                * args.overshooting_kl_beta
                * torch.max(
                    (
                        kl_divergence(
                            Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)),
                            Normal(prior_means, prior_std_devs),
                        )
                        * seq_mask
                    ).sum(dim=2),
                    free_nats,
                ).mean(dim=(0, 1))
                * (args.chunk_size - 1)
            )  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
            # Calculate overshooting reward prediction loss with sequence mask
            if args.overshooting_reward_scale != 0:
                reward_loss += (
                    (1 / args.overshooting_distance)
                    * args.overshooting_reward_scale
                    * F.mse_loss(
                        bottle(reward_model, (beliefs, prior_states)) * seq_mask[:, :, 0],
                        torch.cat(overshooting_vars[2], dim=1),
                        reduction='none',
                    ).mean(dim=(0, 1))
                    * (args.chunk_size - 1)
                )  # Update reward loss (compensating for extra average over each overshooting/open loop sequence)
        
        # Apply linearly ramping learning rate schedule
        if args.learning_rate_schedule != 0:
            for group in model_optimizer.param_groups:
                group['lr'] = min(
                    group['lr'] + args.model_learning_rate / args.model_learning_rate_schedule, args.model_learning_rate
                )
        
        model_loss = observation_loss + reward_loss + kl_loss + cost_loss
        # Update model parameters
        model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
        model_optimizer.step()
        
        ## CBF-Dreamer implementation: Jointly train the Barrier Certificate and Controller

        #  Retrieve imageined trajectories
        with torch.no_grad():
            bc_states = posterior_states.detach()
            bc_beliefs = beliefs.detach()
        # print(bc_beliefs.size())

        with FreezeParameters(model_modules):
            imagination_traj = imagine_ahead(
                bc_states, bc_beliefs, controller, transition_model, args.planning_horizon
            )
            
        imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj
        # print(imged_prior_states.size())
        
        # Calculate the Barrier loss and Controller loss and update the barrier_model and controller
        # Retrieve imageined safety costs pred
        with FreezeParameters(model_modules):
            imged_cost = bottle(cost_model, (imged_beliefs, imged_prior_states))
                
        with FreezeParameters(model_modules + value_model.modules):
            imged_reward = bottle(reward_model, (imged_beliefs, imged_prior_states))
            value_pred = get_through_NN(value_model, imged_prior_states)

        returns = lambda_return(
            imged_reward, value_pred, bootstrap=value_pred[-1], discount=args.discount, lambda_=args.disclam
        )

        imged_barrier = get_through_NN(barrier_model, imged_prior_means)

        # Get the Barrier Loss and Controller Loss and back-propagate to the barrierNN and controller
        controller_return = - torch.mean(torch.sum(returns, dim=0))   
        barrier_return = barrier_loss_stoch_return(imged_cost, imged_barrier, args.cost_threshold, 0.05)
        barrier_loss = torch.mean(torch.sum(barrier_return, dim=0))            
        controller_loss = controller_return + args.eta * barrier_loss
        cbf_optimizer.zero_grad()
        controller_loss.backward()
        nn.utils.clip_grad_norm_(cbf_params_list, args.grad_clip_norm, norm_type=2)    
        cbf_optimizer.step()
        

        # CBF-Dreamer implementation: value loss calculation and optimization
        # Value function network training 
        with torch.no_grad():
            value_beliefs = imged_beliefs.detach()
            value_prior_states = imged_prior_states.detach()
            target_return = returns.detach()
        value_dist = Normal(
            get_through_NN(value_model, value_prior_states), 1
        )  # detach the input tensor from the transition network.
        value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))
        # Update model parameters
        value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm, norm_type=2)
        value_optimizer.step()

        # # Store (0) observation loss (1) reward loss (2) KL loss (3) actor loss (4) value loss
       
        losses.append(
            [observation_loss.item(), reward_loss.item(), cost_loss.item(), kl_loss.item(), barrier_loss.item(), controller_loss.item(), value_loss.item()]
        )

    # Update and plot loss metrics
    losses = tuple(zip(*losses))
    metrics['observation_loss'].append(losses[0])
    metrics['reward_loss'].append(losses[1])
    metrics['cost_loss'].append(losses[2])
    metrics['kl_loss'].append(losses[3])
    metrics['barrier_loss'].append(losses[4])
    metrics['controller_loss'].append(losses[5])
    metrics['value_loss'].append(losses[6])
    # lineplot(metrics['episodes'][-len(metrics['observation_loss']) :], metrics['observation_loss'], 'observation_loss', results_dir)
    # lineplot(metrics['episodes'][-len(metrics['reward_loss']) :], metrics['reward_loss'], 'reward_loss', results_dir)
    # lineplot(metrics['episodes'][-len(metrics['kl_loss']) :], metrics['kl_loss'], 'kl_loss', results_dir)
    # lineplot(metrics['episodes'][-len(metrics['cost_loss']) :], metrics['cost_loss'], 'cost_loss', results_dir)
    # lineplot(metrics['episodes'][-len(metrics['barrier_loss']) :], metrics['barrier_loss'], 'barrier_loss', results_dir)
    # lineplot(metrics['episodes'][-len(metrics['controller_loss']) :], metrics['controller_loss'], 'controller_loss', results_dir)
    # lineplot(metrics['episodes'][-len(metrics['value_loss']) :], metrics['value_loss'], 'value_loss', results_dir)

    # Data collection
    print("Data collection")
    with torch.no_grad():
        observation, total_reward, total_costs = env.reset(), 0, 0
        belief, posterior_state, action = (
            torch.zeros(1, args.belief_size, device=args.device),
            torch.zeros(1, args.state_size, device=args.device),
            torch.zeros(1, env.action_size, device=args.device),
        )
        pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        for t in pbar:
            # print("step",t)
            belief, posterior_state, action, next_observation, reward, done, cost = update_belief_and_act(
                args,
                env,
                controller,
                transition_model,
                encoder,
                belief,
                posterior_state,
                action,
                observation.to(device=args.device),
                explore = True
            )
            D.append(observation, action.cpu(), reward, done, cost)
            total_reward += reward
            total_costs += cost
            observation = next_observation
            # if args.render:
            #     env.render()
            if done:
                pbar.close()
                break

        # Update and plot train reward metrics
        metrics['steps'].append(t + metrics['steps'][-1])
        metrics['episodes'].append(episode)
        metrics['train_rewards'].append(total_reward)
        metrics['train_costs'].append(total_costs)
        
        lineplot(
            metrics['episodes'][-len(metrics['train_rewards']) :],
            metrics['train_rewards'],
            'train_rewards',
            results_dir,
        )
        lineplot(
            metrics['episodes'][-len(metrics['train_costs']) :],
            metrics['train_costs'],
            'train_costs',
            results_dir,
        )

    # Test model
    print("Test model")
    if episode % args.test_interval == 0:
        # Set models to eval mode
        transition_model.eval()
        observation_model.eval()
        reward_model.eval()
        cost_model.eval()
        encoder.eval()
        barrier_model.eval()
        controller.eval()
        value_model.eval()
        # Initialise parallelised test environments
        test_envs = EnvBatcher(
            Env,
            (args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth),
            {},
            args.test_episodes,
        )

        with torch.no_grad():
            observation, total_rewards, total_costs, video_frames, total_violation = test_envs.reset(), np.zeros((args.test_episodes,)), np.zeros((args.test_episodes,)), [], 0
            belief, posterior_state, action = (
                torch.zeros(args.test_episodes, args.belief_size, device=args.device),
                torch.zeros(args.test_episodes, args.state_size, device=args.device),
                torch.zeros(args.test_episodes, env.action_size, device=args.device),
            )
            pbar = tqdm(range(args.max_episode_length // args.action_repeat))
            for t in pbar:
                belief, posterior_state, action, next_observation, reward, done, cost = update_belief_and_act(
                    args,
                    test_envs,
                    controller,
                    transition_model,
                    encoder,
                    belief,
                    posterior_state,
                    action,
                    observation.to(device=args.device),
                    explore=False
                )
                total_rewards += reward.numpy()
                total_costs += cost.numpy()
                for i in cost.numpy():
                    if i > args.cost_threshold:
                        total_violation += 1
                    else:
                        pass
                
                if not args.symbolic_env and episode % args.video_interval == 0:  # Collect real vs. predicted frames for video
                    video_frames.append(
                        make_grid(
                            torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3) + 0.5,
                            nrow=5,
                        ).numpy()
                    )  # Decentre
                observation = next_observation
                if done.sum().item() == args.test_episodes:
                    pbar.close()
                    break

        # Update and plot reward metrics (and write video if applicable) and save metrics
        metrics['average_violation'].append(total_violation/args.test_episodes)
        metrics['test_episodes'].append(episode)
        metrics['test_rewards'].append(total_rewards.tolist())
        metrics['test_costs'].append(total_costs.tolist())
        lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
        lineplot(metrics['test_episodes'][-len(metrics['average_violation']) :], metrics['average_violation'], 'average_violation', results_dir)
        lineplot(metrics['test_episodes'], metrics['test_costs'], 'test_costs', results_dir)
        lineplot(
            np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1],
            metrics['test_rewards'],
            'test_rewards_steps',
            results_dir,
            xaxis='step',
        )
        lineplot(
            np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1],
            metrics['test_costs'],
            'test_costs_steps',
            results_dir,
            xaxis='step',
        )
        if not args.symbolic_env and episode % args.video_interval == 0:
            episode_str = str(episode).zfill(len(str(args.episodes)))
            write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
            save_image(
                torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str)
            )
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Set models to train mode
        transition_model.train()
        observation_model.train()
        reward_model.train()
        cost_model.train()
        barrier_model.train()
        controller.train()
        encoder.train()
        value_model.train()
        # Close test environments
        test_envs.close()

    writer.add_scalar("train_reward", metrics['train_rewards'][-1], metrics['steps'][-1])
    writer.add_scalar("train_cost", metrics['train_costs'][-1], metrics['steps'][-1])    
    writer.add_scalar("train/episode_reward", metrics['train_rewards'][-1], metrics['steps'][-1] * args.action_repeat)
    writer.add_scalar("observation_loss", metrics['observation_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("reward_loss", metrics['reward_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("cost_loss", metrics['cost_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("kl_loss", metrics['kl_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("barrier_loss", metrics['barrier_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("controller_loss", metrics['controller_loss'][0][-1], metrics['steps'][-1])
    # writer.add_scalar("actor_loss", metrics['actor_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("value_loss", metrics['value_loss'][0][-1], metrics['steps'][-1])
    print(
        "episodes: {}, total_steps: {}, train_reward: {}, train_cost: {}".format(
            metrics['episodes'][-1], metrics['steps'][-1], metrics['train_rewards'][-1], metrics['train_costs'][-1]
        )
    )

    # Checkpoint models
    if episode % args.checkpoint_interval == 0:
        torch.save(
            {
                'transition_model': transition_model.state_dict(),
                'observation_model': observation_model.state_dict(),
                'reward_model': reward_model.state_dict(),
                'encoder': encoder.state_dict(),
                'cost_model': cost_model.state_dict(),
                'barrier_model': barrier_model.state_dict(),
                'controller': controller.state_dict(),
                'value_model': value_model.state_dict(),
                'model_optimizer': model_optimizer.state_dict(),
                'cbf_optimizer': cbf_optimizer.state_dict(),
                'value_optimizer': value_optimizer.state_dict(),
            },
            os.path.join(results_dir, 'models_%d.pth' % episode),
        )
        if args.checkpoint_experience:
            torch.save(
                D, os.path.join(results_dir, 'experience.pth')
            )  # Warning: will fail with MemoryError with large memory sizes


# Close training environment
env.close()