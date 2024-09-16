import torch
import os
from src.agents.dispatching_decision_transformer.mingpt import dt_model, utils

#Loads a trained Dispatchin Decison Transformer model
class DispatchingDecisionTransformer():
    default_path = ""

    def __init__(self, model_path=default_path, context_length=6):
        print(os.getcwd())
        self.model_conf = dt_model.GPTConfig(6, context_length, n_layer=6, n_head=8, n_embd=128, max_timestep=35, observation_size=57)
        self.model_dt = dt_model.GPT(self.model_conf)
        self.model_dt.load_state_dict(torch.load("./trained-models/"+model_path, map_location=torch.device('cpu'))) #loads pytorch model from given model_path

    @torch.no_grad()
    def solve_environment(self, environment, lower_bound_factor=1): #Solves the given envrionment with the loaded model
        device = "cpu"
        agent = self.model_dt
        agent.eval()

        #Runs first prediction of action when no sequencce exists yet
        state, reward_sum, done = environment.reset(), 0, False
        state = torch.as_tensor(state)
        action_mask = environment.get_action_mask()
        lower_bound_makespan = environment.get_instance_lower_bound_makespan() # retrieves the lowerbound of the current instance.
        state = state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0) # State is required in format: [[S]]
        returns_to_go = [lower_bound_makespan * lower_bound_factor]
        # first state is from environment, first rtg is target return, and first timestep is 0
        sampled_action = utils.sample(agent, state, 1, temperature=1.0, sample=True, actions=None,
                                      rtgs=torch.tensor(returns_to_go, dtype=torch.long).to(device).unsqueeze(
                                          0).unsqueeze(-1),
                                      timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device),
                                      action_mask=torch.as_tensor(action_mask, dtype=torch.bool).to(device))
        j = 0
        all_states = state
        actions = []
        while not done:
            action = sampled_action.cpu().numpy()[0, -1]
            actions += [sampled_action]
            state, reward, done, action_mask = environment.step(action) #Apply last predicted action
            action_mask = action_mask["mask"]
            state = torch.as_tensor(state)
            j += 1
            if done:
                makespan = environment.makespan
            else:
                state = state.unsqueeze(0).unsqueeze(0).to(device)
                all_states = torch.cat([all_states, state], dim=1)
                returns_to_go += [returns_to_go[-1] - reward] #appends new rtg to sequence of rtgs
                # all_states has all previous states and returns_to_go has all previous rtgs context length is handled in utils.sample
                # timestep is just current timestep
                sampled_action = utils.sample(agent, all_states, 1, temperature=1.0, sample=True,    #Predict next action.
                                              actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(
                                                  1).unsqueeze(0),
                                              rtgs=torch.tensor(returns_to_go, dtype=torch.long).to(device).unsqueeze(
                                                  0).unsqueeze(-1),
                                              timesteps=(min(j, self.model_conf.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(
                                                  device)),
                                              action_mask=torch.as_tensor(action_mask, dtype=torch.bool).to(device))
        return makespan, actions
