import os
import torch
from lib.mingpt import dt_model, utils
class DecisionTransformer():
    def __init__(self,dt_model_name, tianshou_obs, rtg_factor = 1):
        self.device = "cpu"
        self.action_sequence = None
        self.reward_sequence = []
        self.reward_sum = 0
        self.state = None
        self.all_states = None
        self.rtg_factor = rtg_factor
        self.return_to_go_sequence = [self.calc_neuroLs_rtg(tianshou_obs)]
        #load DT model
        self.model_conf = dt_model.GPTConfig(vocab_size=10, block_size=50, n_layer=6, n_head=8, n_embd=128, max_timestep=199, observation_size=128)
        self.model_dt = dt_model.GPT(self.model_conf)
        print(os.getcwd())
        self.model_dt.load_state_dict(torch.load("/NeuroLS_DecisionTransformer/lib/trained_models/" + dt_model_name , map_location=torch.device('cpu'))) #Need to set to your absolute path
        self.agent = self.model_dt.to(self.device)
        self.agent.eval()

    def calc_neuroLs_rtg(self,tianshou_obs):
        lower_bound = tianshou_obs.instance_lower_bound[0]
        initial_makespan = tianshou_obs["meta_features"][0][2].item()
        return_to_go = initial_makespan - lower_bound
        return return_to_go * self.rtg_factor

    def update_rewards(self, reward):
        self.return_to_go_sequence += [self.return_to_go_sequence[-1] - reward]
        self.reward_sequence += [reward]
        self.reward_sum += reward
    def update_states(self, state):
        state = torch.as_tensor(state)
        self.state = state.unsqueeze(0).unsqueeze(0).to(self.device)
        if (self.all_states is None):
            self.all_states = self.state
        else:
            self.all_states = torch.cat([self.all_states, self.state.type(torch.float32).to(self.device)],dim=1)
    def get_sampled_action(self,step):
        # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
        # timestep is just current timestep
        if self.action_sequence is None:
            actions = None
            self.action_sequence = []
        else:
            actions =  torch.tensor(self.action_sequence, dtype=torch.long).to(self.device).unsqueeze(
                1).unsqueeze(0)
        sampled_action = utils.sample(self.agent, self.all_states.to(self.device), 1, temperature=1.0, sample=True,
                                      actions=actions,
                                      rtgs=torch.tensor(self.return_to_go_sequence, dtype=torch.float64).to(
                                          self.device).unsqueeze(
                                          0).unsqueeze(-1),
                                      timesteps=(min(step, 200) * torch.ones((1, 1, 1), dtype=torch.int64).to( #200 for 200 interations
                                          self.device)))

        action = sampled_action.cpu().numpy()[0, -1]
        self.action_sequence += [action]
        return sampled_action.cpu().numpy()[0]

    def reset(self,tianshou_obs):
        #Resets for next instance
        self.action_sequence = None
        self.reward_sequence = []
        self.state = None
        self.all_states = None
        self.return_to_go_sequence = [self.calc_neuroLs_rtg(tianshou_obs)]

