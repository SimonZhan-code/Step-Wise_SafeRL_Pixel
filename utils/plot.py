import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
import plotly
import torch
import plotly.express as px
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import json 

epoch = np.array([i*10 for i in range(1,101)])

def get_model_free_baseline(file_name, method_name, Env):
    # datasets = []
    # fields = ['AverageEpRet', 'AverageEpCost']
    f = open(file_name)
    data = json.load(f)
    # print(exp_data)
    performance_baseline = data[Env][method_name][0]
    cost_baseline = data[Env][method_name][1]
    # ep = exp_data['Epoch']
    dataset = {'method': method_name, 'mean_test_rewards': performance_baseline, 'mean_test_costs': cost_baseline}
    return dataset


def get_lambda_datasets(file_name):
    datasets = pd.read_csv(file_name)
    mean_costs = datasets['mean_sum_costs_median']
    mean_rewards = datasets['objectives_median']
    steps = datasets['timesteps']
    mean_test_costs = []
    mean_test_rewards = []
    interaction = []
    # ep = []
    for i, (step, mean_reward, mean_cost) in enumerate(zip(steps, mean_rewards, mean_costs)):
        if step % 10000 == 0:
            mean_test_costs.append(mean_cost)
            mean_test_rewards.append(mean_reward)
            interaction.append(step)
        else:
            pass  
    data = {'mean_test_rewards': mean_test_rewards, 'mean_test_costs': mean_test_costs}
    dataset = pd.DataFrame.from_dict(data)
    return dataset


def get_slac_datasets(file_name):
    try:
        exp_data = pd.read_csv(file_name)
    except:
        print('Could not read from %s'%file_name)
    # print(exp_data)
    cost = exp_data['mean_cost'][:100]
    cost_std = exp_data['std_cost'][:100]
    performance = exp_data['mean_return'][:100]
    performance_std = exp_data['std_return'][:100]
    data = {'mean_test_rewards': performance, 'mean_test_costs': cost, 'cost_std': cost_std, 'reward_std': performance_std}
    dataset = pd.DataFrame.from_dict(data)
    return dataset


def get_ours_datasets(filename):
    datasets = torch.load(filename)
    cbf_rewards = [np.median(x) for x in datasets['test_rewards']]
    cbf_rewards_std = [np.std(x) for x in datasets['test_rewards']]
    cbf_costs = [np.median(z) for z in datasets['test_costs']]
    cbf_costs_std = [np.std(z) for z in datasets['test_costs']]
    cbf_steps = np.asarray(datasets['steps'])[np.asarray(datasets['test_episodes']) - 1]
    data = {'mean_test_rewards': cbf_rewards, 'mean_test_costs': cbf_costs, 'interact_times': cbf_steps, 'reward_std': cbf_rewards_std, 'cost_std': cbf_costs_std}
    dataset = pd.DataFrame.from_dict(data)
    return dataset



def plot_func(datasets, baselines, title, compare_spec, std_spec, xaxis='Episodes'):
    data = []
    obj = list(datasets.keys())
    colors = [ 
        '#1f77b4',  # muted blue
        '#808080',  # grey
        '#e377c2',  # raspberry yogurt pink
    ]
    transparent_colors = [
        'rgba(173,216,230, 0.5)',  # light blue
        'rgba(119,136,153, 0.5)',  # light grey
        'rgba(255,182,193, 0.5)',  # ligth pink
    ]
    
    for (dataset, c) in zip(obj, range(len(colors))):  
        ys_mean = datasets[dataset][compare_spec]
        ys_std = datasets[dataset][std_spec]
        ys_upper, ys_lower = ys_mean+ys_std, ys_mean-ys_std
        trace_upper = Scatter(x=epoch, y=ys_upper, line=Line(color='rgba(0, 0, 0, 0)'), name='+1 Std. Dev.', showlegend=False)
        trace_mean = Scatter(x=epoch, y=ys_mean, fill='tonexty', fillcolor=transparent_colors[c], line=Line(color=colors[c]), name=dataset)
        trace_lower = Scatter(x=epoch, y=ys_lower, fill='tonexty', fillcolor=transparent_colors[c],line=Line(color='rgba(0, 0, 0, 0)'), name='-1 Std. Dev.', showlegend=False)
        data.extend([trace_upper, trace_mean, trace_lower])
    
    for baseline in list(baselines.keys()):
        y_data = baselines[baseline][compare_spec]
        y_data = np.ones_like(epoch)*y_data
        baseline_plot = Scatter(x=epoch, y=y_data, line=Line(dash='dash'), name=baseline+'_1M_env_steps')
        data.append(baseline_plot)

    plotly.offline.plot(
        {'data': data, 'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})},
        filename=title + '.html',
        auto_open=False,
    )


def main():
    model_free_filename = ["cpo", "trpo_lagrangian", "ppo"]
    env_name = 'PointGoal1'
    baseline_filename = 'baseline_short_results.json'
    lambda_filename = 'lambda_PointGoal1.csv'
    slac_filename = 'slac_PointGoal1.csv'
    ours_filename = 'PointGoal1.pth'
    datasets_baseline = {}
    datasets_mb = {}
      
    for method in model_free_filename:
        datasets_baseline[method] = get_model_free_baseline(baseline_filename, method, env_name)
    # print(datasets)
    # datasets_mb[ours_filename.replace('.pth', '_ours')] = get_ours_datasets(ours_filename)  
    # datasets_mb[lambda_filename.replace('.csv', ' ')] = get_lambda_datasets(lambda_filename)
    datasets_mb[slac_filename.replace('.csv', '')] = get_slac_datasets(slac_filename)

    # plot the graph
    plot_func(datasets_mb, datasets_baseline, 'Rewards_PointGoal1', 'mean_test_rewards', 'reward_std')
    plot_func(datasets_mb, datasets_baseline, 'Costs_PointGoal1', 'mean_test_costs', 'cost_std')






if __name__ == '__main__':
    main()
    