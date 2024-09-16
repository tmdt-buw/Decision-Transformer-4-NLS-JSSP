"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
from src.environments.environment_loader import EnvironmentLoader
from torch.utils.data.dataloader import DataLoader
from src.agents.dispatching_decision_transformer.mingpt.utils import sample
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import math
import logging
import numpy as np
import torch
import torch.optim as optim
import torch
import wandb

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader
    env_config = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'

        if torch.cuda.is_available():
            print("Cuda will be used")
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        print("Cuda will not be used")
        def save_checkpoint(self):
            # DataParallel wrappers keep raw model object in .module attribute
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            logger.info("saving %s", self.config.ckpt_path)
            torch.save(model.state_dict(), "trained_model.pt")
            # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = (split == 'train')
            model.train(True)
            data = self.train_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader))
            print(len(loader))
            for it, (state, action, returns_to_go, timesteps, targets, action_masks) in pbar:
                # place data on the correct device
                states = state.to(self.device)
                actions = action.to(self.device)
                returns_to_go = returns_to_go.to(self.device)
                timesteps = timesteps.to(self.device)
                action_masks = action_masks.to(self.device)

                # forward the model
                with torch.set_grad_enabled(True):
                    logits, loss = model(states, targets, targets, returns_to_go, timesteps, action_masks)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:
                    # backprop and update the parameters
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

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            return losses


        best_mean_makespan = float('inf')
        wb = False
        if wb:
            wandb.init(
            project="ma-wolz",
            config={
                "Datensatz": len(self.train_dataset)
            }
        )
        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            loss = run_epoch('train', epoch_num=epoch)
            loss_avg = np.mean(loss)
            mean_makespan= self.test_model()
            print(mean_makespan)
            if wb:
                wandb.log({"loss": loss_avg})
                wandb.log({"mean_makespan": mean_makespan})
            if mean_makespan < best_mean_makespan: #only save new model if achieved makespan is lower than current best makespan
                best_mean_makespan = mean_makespan
                torch.save(raw_model.state_dict(), "trained_model.pt")
                #wandb.save('trained_model.pt')

    @torch.no_grad()
    def test_model(self):
        eval_makespan = []
        env, _ = EnvironmentLoader.load(self.config.env_config, data=self.test_dataset)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.model.train(False)

        #solve 10 instances with the current model state and return mean makespan
        for i in range(10):
            state, reward_sum, done = torch.as_tensor(env.reset()), 0, False
            lower_bound_makespan = env.get_instance_lower_bound_makespan()
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            returns_to_go = [lower_bound_makespan]
            action_mask = torch.ones(env.num_jobs)
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(raw_model, state, 1, temperature=1.0, sample=True, actions=None,
                                    rtgs=torch.tensor(returns_to_go, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1),
                                    timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device), action_mask=torch.as_tensor(action_mask, dtype=torch.bool).to(self.device))
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
                    eval_makespan.append(env.makespan)
                else:
                    state = state.unsqueeze(0).unsqueeze(0).to(self.device)
                    all_states = torch.cat([all_states, state], dim=1)
                    returns_to_go += [returns_to_go[-1] - reward]
                    # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                    # timestep is just current timestep
                    sampled_action = sample(raw_model, all_states, 1, temperature=1.0, sample=True,
                                            actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0),
                                            rtgs=torch.tensor(returns_to_go, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1),
                                            timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1),dtype=torch.int64).to(self.device)), action_mask=torch.as_tensor(action_mask, dtype=torch.bool).to(self.device))
            print("target return: %d, eval return: %d, diff: %d" % (lower_bound_makespan, eval_makespan[-1], (lower_bound_makespan+eval_makespan[-1])))
        self.model.train(True)
        mean_makespan = np.mean(eval_makespan)
        return mean_makespan
