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


def get_model_free_datasets(file_name, condition=None):
    # datasets = []
    # fields = ['AverageEpRet', 'AverageEpCost']
    try:
        exp_data = pd.read_table(file_name)
    except:
        print('Could not read from %s'%file_name)
    # print(exp_data)
    cost = exp_data['AverageEpCost']
    performance = exp_data['AverageEpRet']
    Steps = exp_data['TotalEnvInteracts']
    data = {'mean_test_rewards': performance, 'mean_test_costs': cost, 'steps': Steps}
    dataset = pd.DataFrame.from_dict(data)
    return dataset


def get_lambda_datasets(file_name):
    datasets = pd.read_csv(file_name)
    mean_costs = datasets['mean_sum_costs_median']
    mean_rewards = datasets['objectives_median']
    steps = datasets['timesteps']
    mean_test_costs = []
    mean_test_rewards = []
    interaction = []
    for i, (step, mean_reward, mean_cost) in enumerate(zip(steps, mean_rewards, mean_costs)):
        if step % 10000 == 0:
            mean_test_costs.append(mean_cost)
            mean_test_rewards.append(mean_reward)
            interaction.append(step)
        else:
            pass
    data = {'mean_test_rewards': mean_test_rewards, 'mean_test_costs': mean_test_costs, 'steps': interaction}
    dataset = pd.DataFrame.from_dict(data)
    return dataset


def get_slac_datasets(file_name):
    datasets = pd.read_csv(file_name)
    mean_costs = datasets['cost']
    mean_rewards = datasets['return']
    step = datasets['step']
    data = {'mean_test_rewards': mean_rewards, 'mean_test_costs': mean_costs, 'steps': step}
    dataset = pd.DataFrame.from_dict(data)
    return dataset


def get_ours_datasets(filename):
    datasets = torch.laod(filename)
    cbf_rewards = [np.mean(x) for x in datasets['test_rewards']]
    cbf_costs = [np.mean(z) for z in datasets['test_costs']]
    data = {'mean_test_rewards': cbf_rewards, 'mean_test_costs': cbf_costs}
    dataset = pd.DataFrame.from_dict(data)
    return dataset


def plot_func(datasets, title, xaxis='Steps'):
    data = []
    obj = list(datasets.keys())
    colors = [ 
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
    ]
    for (dataset, c) in zip(obj, colors):
        data.append(Scatter(x=datasets[dataset]['steps'], y=datasets[dataset][title], line=Line(color=c), name=dataset))
    plotly.offline.plot(
        {'data': data, 'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})},
        filename=title + '.html',
        auto_open=False,
    )


def main():
    model_free_filename = ['cpo_PointButton1.txt', 'trpo_PointButton1.txt', 'ppo_PointButton1.txt']
    lambda_filename = 'lambda_PointButton1.csv'
    slac_filename = 'slac_PointButton1.csv'
    datasets = {}
    for file_name in model_free_filename:
        datasets[file_name.replace('.txt', ' ')] = get_model_free_datasets(file_name)
    # print(datasets)
    datasets[lambda_filename.replace('.csv', ' ')] = get_lambda_datasets(lambda_filename)
    datasets[slac_filename.replace('.csv', ' ')] = get_slac_datasets(slac_filename)
    # print(get_slac_datasets(slac_filename))
    # steps = np.array([i * 10000 for i in range(1, 51)])
    plot_func(datasets, 'mean_test_rewards')
    plot_func(datasets, 'mean_test_costs')
    




if __name__ == '__main__':
    main()
    