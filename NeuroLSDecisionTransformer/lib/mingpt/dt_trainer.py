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
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import math
import logging
import numpy as np
import torch
import torch.optim as optim
import torch
import wandb
import subprocess as sp
import os

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
            for it, (state, action, returns_to_go, timesteps, targets) in pbar:
                # place data on the correct device
                states = state.to(self.device)
                actions = action.to(self.device)
                returns_to_go = returns_to_go.to(self.device)
                timesteps = timesteps.to(self.device)
                #action_masks = action_masks.to(self.device)

                # forward the model
                with torch.set_grad_enabled(True):
                    logits, loss = model(states, targets, targets, returns_to_go, timesteps)
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
        wb = True
        if wb:
            wandb.init(
            project="ma-wolz",
            config={
                "Datensatz": len(self.train_dataset)
            }
        )
        self.tokens = 0  # counter used for learning rate decay
        os.chdir("../")
        for epoch in range(config.max_epochs):
            out = None
            loss = run_epoch('train', epoch_num=epoch)
            mean_loss = np.mean(loss)
            if epoch % 20 == 0: #Only run test every 20 steps
                torch.save(model.module.state_dict(),
                           os.path.join(os.getcwd(),
                                        "/NeuroLS_DecisionTransformer/trained_model_nls.pt"))
                path = os.path.join(os.getcwd(), "/NeuroLS_DecisionTransformer/run_benchmark.py")
                path2 = os.path.join(os.getcwd(), "/NeuroLS_DecisionTransformer/run_nls_jssp.py")
                path3 = os.path.join(os.getcwd(), "/NeuroLS_DecisionTransformer/data/JSSP/jssp20x20/")
                cmd = 'python ' + path + ' -r ' + path2 + ' -d ' + path3 + ' -g Validation -p jssp -m nls -e eval_jssp --args env=jssp20x20_unf -n 200'
                try:
                    print(os.getcwd())
                    out = sp.run(cmd.split(),
                                 universal_newlines=True,
                                 capture_output=True,
                                 check=True
                                 )
                    print(out.stdout)
                except sp.CalledProcessError as e:
                    print(f"encountered error for call: {e.cmd}\n")
                    print(e.stderr)
                makespan = float(out.stdout.split("makespan")[1])
                if wb:
                    wandb.log({"loss": mean_loss})
                    if out is not None:
                        wandb.log({"mm": makespan})
                if makespan < best_mean_makespan + 5:
                    best_mean_makespan = makespan
                    torch.save(model.module.state_dict(), os.path.join(os.getcwd(),
                                                                       "wolz/projects/NeuroLS_DecisionTransformer/trained_model_nls_best.pt"))
                    wandb.save("wolz/projects/NeuroLS_DecisionTransformer/trained_model_nls_best.pt")
            else:
                print(mean_loss)
                if wb:
                    wandb.log({"loss": mean_loss})
