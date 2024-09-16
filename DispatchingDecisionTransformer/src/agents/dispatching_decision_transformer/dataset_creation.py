"""creates the datasets for the decision transformer"""
import os
from asyncio import Task

import yaml
import pickle
import torch as T
import numpy as np
from typing import List
from pathlib import Path

from src.environments.observation_parser import ObservationParser
from src.agents.reinforcement_learning.ppo_masked import MaskedPPO
from src.environments.environment_loader import EnvironmentLoader
from src.agents.dispatching_decision_transformer.utils.loading_utils import load_config, load_data_splits
from src.agents.train_test_utility_functions import load_data

# ___ CONSTANTS ___
CONFIG_PATH = Path('./base_networks/6x6_agent_mppo_config.yaml')
MODEL_PATH = '../dispatching_decision_transformer/base_networks/6x6_agent_mppo'

def play_through_data_with_other_observation(agent, environment, data, deterministic):
    """
    Plays through the data and stores tuples of:
    - the instance hash
    - the step number
    - the observation taken in that step
    - the action taken in that step
    - the action logits in that step
    - the action probabilities in that step
    - the reward received for taking that action in the step
    - action mask
    :param agent: trained agent that maps observations to actions
    :param environment: environment which is used to play through the data
    :param data: problem instances that are played through
    :param deterministic: whether the agent should act deterministically
    :return: list filled with the dictionaries
    """
    data_points = []
    returns_to_go = []
    observation_strategys = ["full_raw", "mtr_comparison"] #Set wanted observation_strategys
    for instance in data:
        runtime = environment.get_instance_lower_bound_makespan()
        hash = instance[0].instance_hash
        # play through the instance
        current_instance = []
        for step in range(len(instance)):
            # get observation
            observation = environment.state_obs
            full_raw_observation = np.concatenate([ObservationParser.parse(environment, obs) for obs in observation_strategys])
            action_mask = environment.get_action_mask()
            step_number = environment.num_steps
            makespan = environment.makespan
            # get action
            prediction_distribution = agent.policy_net(T.tensor(observation).float(), T.tensor(action_mask).bool())
            logits = prediction_distribution.logits.detach().numpy()
            probabilities = prediction_distribution.probs.detach().numpy()
            if deterministic:
                action = T.argmax(prediction_distribution.probs).detach().numpy().item()
            else:
                action = prediction_distribution.sample().detach().numpy().item()
            # get reward
            _, reward, done, infos = environment.step(action)
            # store tuple
            data_point = {'instance_hash': hash,
                          'step_number': step_number,
                          'observation': full_raw_observation,
                          'action': action,
                          'logits': logits,
                          'probabilities': probabilities,
                          'reward': reward,
                          'makespan': makespan,
                          'action_mask': action_mask,
                          'returns_to_go': None,
                          'lowerbound': env.get_instance_lower_bound_makespan()
                          }
            current_instance.append(data_point)
            if done:
                returns_to_go.extend(calculate_returns_to_go(current_instance))
                data_points.extend(current_instance)
    return data_points


def play_through_data(agent, environment, data, deterministic):
    """
    Plays through the data and stores tuples of:
    - the instance hash
    - the step number
    - the observation taken in that step
    - the action taken in that step
    - the action logits in that step
    - the action probabilities in that step
    - the reward received for taking that action in the step
    - action mask
    :param agent: trained agent that maps observations to actions
    :param environment: environment which is used to play through the data
    :param data: problem instances that are played through
    :param deterministic: whether the agent should act deterministically
    :return: list filled with the dictionaries
    """
    data_points = []
    returns_to_go = []
    for instance in data:
       # makespan_lower_bound = get_makespan_lower_bound(data)
        # reset environment
        environment.reset()
        runtime = environment.get_instance_lower_bound_makespan()
        hash = instance[0].instance_hash
        # play through the instance
        current_instance = []
        for step in range(len(instance)):
            # get observation
            observation = environment.state_obs
            action_mask = environment.get_action_mask()
            step_number = environment.num_steps
            makespan = environment.makespan
            # get action
            prediction_distribution = agent.policy_net(T.tensor(observation).float(), T.tensor(action_mask).bool())
            logits = prediction_distribution.logits.detach().numpy()
            probabilities = prediction_distribution.probs.detach().numpy()
            if deterministic:
                action = T.argmax(prediction_distribution.probs).detach().numpy().item()
            else:
                action = prediction_distribution.sample().detach().numpy().item()
            _, reward, done, infos = environment.step(action)
            # store tuple
            data_point = {'instance_hash': hash,
                          'step_number': step_number,
                          'observation': observation,
                          'action': action,
                          'logits': logits,
                          'probabilities': probabilities,
                          'reward': reward,
                          'makespan': makespan,
                          'action_mask': action_mask,
                          'returns_to_go': None,
                          'lowerbound': env.get_instance_lower_bound_makespan()
                          }
            current_instance.append(data_point)
            if done:
                returns_to_go.extend(calculate_returns_to_go(current_instance))
                data_points.extend(current_instance)
    return data_points


def play_through_data_optimal(data):
    # Creates dataset with trajectories from the optimal solver
    import src.agents.solver.solver as Solver
    solver = Solver.OrToolSolver()
    data_points = []
    for instance in data:
        returns_to_go = []
        current_instance = []
        hash = instance[0].instance_hash
        environment, _ = EnvironmentLoader.load(config, data=[instance])

        #need to derive the action from optimal solutions so we can solve the environment with the actions to get rewards observations and action_masks from environtment
        solved, optimal_makespan = solver.optimize(instance)
        solved_list = [element for x in solved for element in solved[x]]
        sorted_by_start_time = sorted(solved_list, key=lambda task: task.start)
        actions = [task.job for task in sorted_by_start_time]

        #use the derived actions from OR solver to solve enironmen
        for action in actions:
            observation = environment.state_obs
            action_mask = environment.get_action_mask()
            step_number = environment.num_steps
            makespan = environment.makespan
            _, reward, done, infos = environment.step(action)

            data_point = {'instance_hash': hash,
                          'step_number': step_number,
                          'observation': observation,
                          'action': action,
                          'reward': reward,
                          'makespan': makespan,
                          'action_mask': action_mask,
                          'returns_to_go': None,
                          'lowerbound': environment.get_instance_lower_bound_makespan()
                          }
            current_instance.append(data_point)
        if done:
            returns_to_go.extend(calculate_returns_to_go(current_instance))
            data_points.extend(current_instance)

    return data_points


def calculate_returns_to_go(datapoints: List[dict]) -> List[int]:
    # Iterates over the datapoints and calculates the rtgs for each datapoint.
    return_to_go = 0
    return_to_go_list = []

    for datapoint in reversed(datapoints): #Iterate reversed because last rtg is zero and first rtg is sum of all rewards achieved
        return_to_go = return_to_go + datapoint.get("reward")
        datapoint['returns_to_go'] = return_to_go
        return_to_go_list.append(return_to_go)
    return_to_go_list.reverse()
    return return_to_go_list


if __name__ == '__main__':
    # load config
    config = load_config()
    train_data, test_data, val_data = load_data_splits(config)
    np.random.seed(config['seed'])
    # create environment
    env, _ = EnvironmentLoader.load(config, data=train_data)
    # Load the specified model
    # creates a model according to the config file. Still has to be overwritten with learned parameters
    model = MaskedPPO(env, config=config)
    model = model.load(MODEL_PATH, config)  # overwrites parameters
    device = 'cpu'
    model.policy_net.device = 'cpu'
    model.value_net.to(device)
    model.policy_net.to(device)

    # play through the data
    data_points = play_through_data(agent=model, environment=env, data=train_data, deterministic=True)
    #data_points = play_through_data_with_other_observation(agent=model, environment=env, data=train_data, deterministic=True)
    #data_points = play_through_data_optimal(data=train_data)

    # save data points in pickle file
    with open('datasets/raw_datasets/data_points_optimal.pkl', 'wb') as f:
        pickle.dump(data_points, f)