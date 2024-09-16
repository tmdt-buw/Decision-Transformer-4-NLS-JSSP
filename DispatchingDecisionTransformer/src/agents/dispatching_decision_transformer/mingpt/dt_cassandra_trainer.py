import os
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist
from src.agents.dispatching_decision_transformer.mingpt.utils import sample
from src.environments.environment_loader import EnvironmentLoader
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np
import torch.optim as optim
import torch
import wandb


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, train_data, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_data = train_data[:100]
        self.config = config
        self.world_size = 8
        os.environ['MASTER_ADDR'] = 'localhost'  #
        os.environ['MASTER_PORT'] = '8123'  #
        self.tokens = 0
        self.wb = True

    def train(self):  # Number of gpus in Cassandra
        mp.spawn(self.parallel_train, args=(self.world_size,), nprocs=self.world_size)

    def parallel_train(self, gpu, world_size):
        dist.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=gpu
        )
        if self.wb and gpu == 0:
            wandb.init(
                project="ma-wolz",
                config={
                    "Datensatz": len(self.train_dataset),
                    "Cassandra Distributed": True,
                    "Batch Size": self.config.batch_size
                }
            )
        model, config = self.model, self.config
        optimizer = model.configure_optimizers(config)
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)
        model.train(True)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=gpu
        )

        train_loader = torch.utils.data.DataLoader(
            dataset= self.train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler)

        total_step = len(train_loader)
        best_mean_makespan = float('inf')
        for epoch in range(config.max_epochs):
            losses = []
            train_loader.sampler.set_epoch(epoch)
            for it, (state, action, returns_to_go, timesteps, action_masks) in enumerate(train_loader):
                # place data on the correct device
                states = state.cuda(non_blocking=True)
                actions = action.cuda(non_blocking=True)
                returns_to_go = returns_to_go.cuda(non_blocking=True)
                timesteps = timesteps.cuda(non_blocking=True)
                action_masks = action_masks.cuda(non_blocking=True)

                with torch.set_grad_enabled(True):
                    logits, loss = model(states, actions, actions, returns_to_go, timesteps, action_masks)
                    losses.append(loss.item())

                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                                actions >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                if gpu == 0:
                    print("TestPrint")
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Learning Rate{}'.format(
                        epoch + 1,
                        config.max_epochs,
                        it + 1,
                        total_step,
                        loss.item(), lr)
                    )
            if gpu == 0:
                mean_loss = np.mean(losses)
                if epoch % 10 == 0:
                    mean_makespan_test = self.test_model(self.test_dataset)
                    mean_makespan_train = self.test_model(self.train_data)
                    print(mean_loss)
                    print(mean_makespan_test)
                    if self.wb:
                        wandb.log({"loss": mean_loss})
                        wandb.log({"mean_makespan": mean_makespan_test})
                        wandb.log({"mean_makespan_train_data": mean_makespan_train})
                        wandb.log({"learning_rate": lr})
                    if mean_makespan_test < best_mean_makespan:
                        best_mean_makespan = mean_makespan_test
                        torch.save(model.module.state_dict(), "trained_model.pt")
                else:
                    print(mean_loss)
                    if self.wb:
                        wandb.log({"loss": mean_loss})
                        wandb.log({"learning_rate": lr})

    @torch.no_grad()
    def test_model(self, dataset):
        device = "cuda:0"
        eval_makespan = []
        env, _ = EnvironmentLoader.load(self.config.env_config, data=dataset)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.model.train(False)

        T_rewards, T_Qs = [], []
        for i in range(11):
            state, reward_sum, done = torch.as_tensor(env.reset()), 0, False
            lower_bound_makespan = env.get_instance_lower_bound_makespan()
            state = state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
            returns_to_go = [lower_bound_makespan]
            action_mask = env.get_action_mask()
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(raw_model, state, 1, temperature=1.0, sample=True, actions=None,
                                    rtgs=torch.tensor(returns_to_go, dtype=torch.long).to(device).unsqueeze(
                                        0).unsqueeze(-1),
                                    timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device),
                                    action_mask=torch.as_tensor(action_mask, dtype=torch.bool).to(device))
            j = 0
            all_states = state
            rewards = []
            actions = []
            while not done:
                action = sampled_action.cpu().numpy()[0, -1]
                actions += [sampled_action]
                state, reward, done, action_mask = env.step(action)
                action_mask = action_mask["mask"]
                rewards += [reward]
                state = torch.as_tensor(state)
                reward_sum += reward
                j += 1
                if done:
                    T_rewards.append(reward_sum)
                    eval_makespan.append(env.makespan)
                else:
                    state = state.unsqueeze(0).unsqueeze(0).to(device)
                    all_states = torch.cat([all_states, state], dim=1)
                    returns_to_go += [returns_to_go[-1] - reward]
                    # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                    # timestep is just current timestep
                    sampled_action = sample(raw_model, all_states, 1, temperature=1.0, sample=True,
                                            actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(
                                                1).unsqueeze(0),
                                            rtgs=torch.tensor(returns_to_go, dtype=torch.long).to(
                                                device).unsqueeze(0).unsqueeze(-1),
                                            timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1),
                                                                                                     dtype=torch.int64).to(
                                                device)),
                                            action_mask=torch.as_tensor(action_mask, dtype=torch.bool).to(device))
            T_rewards = []
            print("target return: %d, eval return: %d, diff: %d" % (
            lower_bound_makespan, eval_makespan[-1], (lower_bound_makespan + eval_makespan[-1])))
        self.model.train(True)
        mean_makespan = np.mean(eval_makespan)
        return mean_makespan





