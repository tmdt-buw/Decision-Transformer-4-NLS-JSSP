from src.agents.reinforcement_learning.ppo_masked import MaskedPPO
import torch
import os
from pathlib import Path
from src.agents.train_test_utility_functions import load_data
import yaml
from sklearn.model_selection import train_test_split

CONFIG_PATH = Path('./base_networks/6x6_agent_mppo_config.yaml')


def load_mppo(config, env, path):
    model = MaskedPPO(env, config=config)
    model = model.load(path, config)  # overwrites parameters
    device = 'cpu'
    model.policy_net.device = 'cpu'
    model.value_net.to(device)
    model.policy_net.to(device)
    return model


def solve_mppo(environment, agent):
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
        # action = np.where(action_mask, np.random.rand(6), -np.inf).argmax()
        actions.append(action)
        # get reward
        state, reward, done, infos = environment.step(action)
        if done:
            return actions, environment.makespan

def load_config():
    os.chdir(Path(__file__).parent.parent.absolute())
    print(os.getcwd())
    with open(CONFIG_PATH, mode='r') as open_file:
        config = yaml.safe_load(open_file)
    config['wandb_mode'] = 0
    # parse config from wandb to our format
    config = {key: value['value'] for (key, value) in config.items() if type(value) == dict}
    # load data
    # random seed for numpy as given by config

    return config

def load_data_splits(config):
    data = load_data(config)
    print(len(data))
    # train/test/validation data split
    split_random_seed = config['seed'] if not config.get('overwrite_split_seed', False) else 1111
    train_data, test_data = train_test_split(
        data, train_size=config.get('train_test_split'), random_state=split_random_seed)
    test_data, val_data = train_test_split(
        test_data, train_size=config.get('test_validation_split'), random_state=split_random_seed)
    print(len(test_data),len(val_data))
    return train_data, test_data, val_data
