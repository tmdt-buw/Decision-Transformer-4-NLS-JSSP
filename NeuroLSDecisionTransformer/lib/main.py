
import pickle
import numpy as np
from mingpt import dt_trainer, dt_model, dt_cassandra_trainer
from lib.utils.StateActionReturnDataset import StateActionReturnDataset

# ___ CONSTANTS ___
CONTEXT_LENGTH = 50
MAX_TIMESTEPS = 199 # number of LS iterations per instace

if __name__ == '__main__':
    np.random.seed(123)

    with open("projects/NeuroLS_DecisionTransformer/lib/data_points_nls_flat.pkl", "rb") as handle: #load dataset created by create_dt_dataset notebook
        train_data = pickle.load(handle)

    train_dataset = StateActionReturnDataset(train_data, CONTEXT_LENGTH)

    mconf = dt_model.GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=6, n_head=8, n_embd=128, max_timestep=MAX_TIMESTEPS, observation_size=128)
    model = dt_model.GPT(mconf)
    test_data = None #Not explicitly need since tests are run via run_benchmark.py where the test_dataset is given as path

    # initialize a trainer instance and kick off training
    tconf = dt_trainer.TrainerConfig(max_epochs=500, batch_size=128, learning_rate=6e-4,
                          lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*CONTEXT_LENGTH,
                          num_workers=4, seed=123, max_timestep=MAX_TIMESTEPS) #max timesteps = number of LS iterations per instace

    # Either load dt_cassandra_trainer for parrallel training on multiple gpus or dt_trainer for single gpu/cpu training.
    trainer = dt_cassandra_trainer.Trainer(model, train_dataset, test_data, tconf)
    trainer.train()
