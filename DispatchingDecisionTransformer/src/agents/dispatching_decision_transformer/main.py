import yaml
import pickle
import numpy as np
from src.agents.dispatching_decision_transformer.mingpt import dt_trainer, dt_model
from src.agents.dispatching_decision_transformer.StateActionReturnDataset import StateActionReturnDataset
from sklearn.model_selection import train_test_split
from src.agents.train_test_utility_functions import load_data

# ___ CONSTANTS ___
CONFIG_PATH = '../dispatching_decision_transformer/base_networks/6x6_agent_mppo_config.yaml'
DDT_TRAINDATA_PATH = "../dispatching_decision_transformer/datasets/raw_datasets/data_points_rtglower_100.pkl"
CONTEXT_LENGTH = 6

if __name__ == '__main__':
    with open(CONFIG_PATH, mode='r') as open_file:
        config = yaml.safe_load(open_file)
    config = {key: value['value'] for (key, value) in config.items() if type(value) == dict}

    data = load_data(config)  # List[List[Task]]
    np.random.seed(config['seed'])

    split_random_seed = config['seed'] if not config.get('overwrite_split_seed', False) else 1111
    train_data, test_data = train_test_split(
        data, train_size=config.get('train_test_split'), random_state=split_random_seed)
    test_data, val_data = train_test_split(
        test_data, train_size=config.get('test_validation_split'), random_state=split_random_seed)

    with open(DDT_TRAINDATA_PATH, "rb") as handle:
        train_data = pickle.load(handle)

    #Initialize StateActionReturnDataset and load GPT model.
    train_dataset = StateActionReturnDataset(train_data, CONTEXT_LENGTH)
    mconf = dt_model.GPTConfig(train_dataset.vocab_size, train_dataset.context_length, n_layer=6, n_head=8, n_embd=128, max_timestep=35, observation_size=57)
    model = dt_model.GPT(mconf)

    # initialize a trainer instance and kick off training
    tconf = dt_trainer.TrainerConfig(max_epochs=100, batch_size=512, learning_rate=6e-4,
                          lr_decay=False, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*CONTEXT_LENGTH,
                          num_workers=4, seed=config['seed'], max_timestep=35, env_config=config)
    trainer = dt_trainer.Trainer(model, train_dataset, test_data, tconf)
    trainer.train()
