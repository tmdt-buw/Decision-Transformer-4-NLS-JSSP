import os
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
import math
import numpy as np
import torch.optim as optim
import torch
import wandb
import subprocess as sp


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.world_size = 2 # Number of gpus
        os.environ['MASTER_ADDR'] = 'localhost'  #
        os.environ['MASTER_PORT'] = '8123'  #
        self.wb = True
        self.tokens = 0

    def train(self):  # Number of gpus in Cassandra
        mp.spawn(self.parallel_train, args=(self.world_size,), nprocs=self.world_size) #runs paarallel_train() on each gpu

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
            dataset=self.train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler)

        total_step = len(train_loader)
        best_mean_makespan = float('inf')
        os.chdir("../")
        for epoch in range(config.max_epochs):
            losses = []
            train_loader.sampler.set_epoch(epoch)
            for it, (state, action, returns_to_go, timesteps, targets) in enumerate(train_loader):
                # place data on the correct device
                states = state.cuda(non_blocking=True)
                actions = action.cuda(non_blocking=True)
                returns_to_go = returns_to_go.cuda(non_blocking=True)
                timesteps = timesteps.cuda(non_blocking=True)

                with torch.set_grad_enabled(True):
                    logits, loss = model(states, actions, actions, returns_to_go, timesteps)
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
            group_losses = [None for _ in range(world_size)]
            dist.all_gather_object(group_losses, np.mean(losses))
            if gpu == 0:
                out = None
                mean_loss = np.mean(group_losses)
                if epoch % 20 == 0: #Only test model every 20 steps
                    torch.save(model.module.state_dict(),
                               os.path.join(os.getcwd(),
                                            "wolz/projects/NeuroLS_DecisionTransformer/trained_model_nls.pt"))
                    path = os.path.join(os.getcwd(), "wolz/projects/NeuroLS_DecisionTransformer/run_benchmark.py")
                    path2 = os.path.join(os.getcwd(), "wolz/projects/NeuroLS_DecisionTransformer/run_nls_jssp.py")
                    path3 = os.path.join(os.getcwd(), "wolz/projects/NeuroLS_DecisionTransformer/data/JSSP/jssp15x15/")
                    cmd = 'python ' + path + ' -r ' + path2 + ' -d ' + path3 + ' -g Validation -p jssp -m nls -e eval_jssp --args env=jssp15x15_unf -n 200' #Change problem size if needed
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
                    if self.wb:
                        wandb.log({"loss": mean_loss})
                        if out is not None:
                            wandb.log({"mm": makespan})
                        wandb.log({"learning_rate": lr})
                    if makespan < best_mean_makespan + 2: #Give some tolerance in model saving
                        best_mean_makespan = makespan
                        torch.save(model.module.state_dict(), os.path.join(os.getcwd(),
                                                                           "wolz/projects/NeuroLS_DecisionTransformer/trained_model_nls_best" + str(
                                                                               int(makespan)) + ".pt"))
                        wandb.save("wolz/projects/NeuroLS_DecisionTransformer/trained_model_nls_best" + str(
                            int(makespan)) + ".pt")
                else:
                    print(mean_loss)
                    if self.wb:
                        wandb.log({"loss": mean_loss})
                        wandb.log({"learning_rate": lr})