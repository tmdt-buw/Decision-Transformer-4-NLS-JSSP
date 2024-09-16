
import copy
from asyncio import Task

import yaml
import pickle
import torch
import numpy as np
from pathlib import Path
from src.agents.dispatching_decision_transformer.mingpt import dt_model,utils

from typing import List
from src.agents.dispatching_decision_transformer.dataset_creation import load_config, load_data_splits
from src.agents.dispatching_decision_transformer.DispatchingDecisionTransformer import DispatchingDecisionTransformer
from src.agents.reinforcement_learning.ppo_masked import MaskedPPO
from sklearn.model_selection import train_test_split
from src.environments.environment_loader import EnvironmentLoader
from src.agents.train_test_utility_functions import load_data

# ___ CONSTANTS ___
MODEL_PATH = '../dispatching_decision_transformer/base_networks/6x6_agent_mppo'


@torch.no_grad()
#Solves the given environment with the given agent and returns achieved mean makespan
def eval_agent(agent, environment, makespans: list):
    for i in range(len(test_data)):
        if type(agent) == DispatchingDecisionTransformer:
            makespans.append(agent.solve_environment(environment)[0])
        else:
            state = environment.reset()
            # play through the instance
            done = False
            actions = []
            while not done:
                action_mask = environment.get_action_mask()
                step_number = environment.num_steps
                # get observation
                prediction_distribution = agent.policy_net(torch.tensor(state).float(), torch.tensor(action_mask).bool())
                action = torch.argmax(prediction_distribution.probs).detach().numpy().item()
                #action = prediction_distribution.sample().detach().numpy().item()  # TODO train with non determinisitc actions
                #action = np.where(action_mask, np.random.rand(6), -np.inf).argmax()
                actions.append(torch.as_tensor(action))
                # get reward
                state, reward, done, infos = environment.step(action)
                if done:
                    makespans.append(environment.makespan)


def solve_optimal(data, makespans):
    import src.agents.solver.solver as Solver
    solver = Solver.OrToolSolver()
    data_points = []
    for instance in data:
        environment, _ = EnvironmentLoader.load(config, data=[instance])
        solved, optimal_makespan = solver.optimize(instance)
        solved_list = [element for x in solved for element in solved[x]]
        sorted_by_start_time = sorted(solved_list, key=lambda task: task.start)
        actions = [task.job for task in sorted_by_start_time]

        for action in actions:
            _, reward, done, infos = environment.step(action)
        if done:
            makespans.append(environment.makespan)



if __name__ == '__main__':
    # load config and data
    config = load_config()
    train_data, test_data, val_data = load_data_splits(config)
    test_data = test_data[:100]
    np.random.seed(1111)

    #Initialize environment for mppo and  with the same test data
    env, _ = EnvironmentLoader.load(config, data=test_data)
    env_dt = copy.deepcopy(env)
    # Load the mppo model
    mppo = MaskedPPO(env, config=config)
    mppo = mppo.load(MODEL_PATH, config)  # overwrites parameters
    device = 'cpu'
    mppo.policy_net.device = 'cpu'
    mppo.value_net.to(device)
    mppo.policy_net.to(device)
    #load decision transfomer with given model and and given context length
    dt = DispatchingDecisionTransformer("CL6.pt", context_length=6)
    #dt = DispatchingDecisionTransformer("CL2.pt", cl=2)
    #dt = DispatchingDecisionTransformer("CL30.pt", cl=6)

    makespan_ppo = []
    makespan_dt = []
    makespan_opt = []
    print(len(test_data))

    eval_agent(dt, env_dt, makespan_dt)
    eval_agent(mppo, env, makespan_ppo)

    print("Mean makespan dt:", np.mean(makespan_dt))
    print("Mean makespan ppo:", np.mean(makespan_ppo))