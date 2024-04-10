import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.algorithms.actor_critic import PPO,DDPG
from mushroom_rl.core import Core
from mushroom_rl.policy import GaussianTorchPolicy,BoltzmannTorchPolicy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter
from environment.simRadar_fre_history_two import SimRadar
from fre_current_two.save_res import save_csv

n_steps = 100
n_steps_per_fit = 10
n_epochs = 300

actionLen = 6
jamType = 0
rewardType = 2
import sys

if len(sys.argv) >= 3:
    actionLen = int(sys.argv[1])
    jamType = int(sys.argv[2])
    rewardType = int(sys.argv[3])
radarStepNum = 10

radarType = "FMCW"

historyLen=8
mdp = SimRadar(actionLen=actionLen , radarStepNum=radarStepNum, jamType=jamType , radarType = radarType,rewardType=rewardType,historyLen=historyLen)

class Network(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super(Network, self).__init__()
        n_features = 32
        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)
        self.attention = nn.Sequential(
            nn.Linear(n_input, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_input),
            nn.Softmax(dim=-1))

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        attention_weights = self.attention(state.float())
        state=torch.mul(state, attention_weights)
        features1 = torch.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = torch.relu(self._h2(features1))
        a = self._h3(features2)

        return a




# MDP
horizon = 500
gamma = 0.99
gamma_eval = 1.
# mdp = DMControl('walker', 'stand', horizon, gamma)


# Policy
# policy_class = OrnsteinUhlenbeckPolicy
# policy_params = dict(sigma=np.ones(1) * .2, theta=.15, dt=1e-2)
# policy_params = dict(std_0=1.)
#
# policy_class = GaussianTorchPolicy(Network,
#                              mdp.info.observation_space.shape,
#                              mdp.info.action_space.shape,
#                              **policy_params)
policy_params = dict(
        n_features=32
    )

beta = Parameter(1e0)
policy_class = BoltzmannTorchPolicy(Network,
                              mdp.info.observation_space.shape,
                              (mdp.info.action_space.n,),
                              beta=beta,
                              **policy_params)

# Agent

params = dict(actor_optimizer={'class': optim.Adam,
                               'params': {'lr': 3e-4}},
              n_epochs_policy=4, batch_size=64, eps_ppo=.2, lam=.95)
critic_params = dict(network=Network,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                        n_features=32,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

params['critic_params'] = critic_params
agent = PPO(mdp.info, policy_class, **params)
# def __init__(self, mdp_info, policy, actor_optimizer, critic_params,
#                  n_epochs_policy, batch_size, eps_ppo, lam, ent_coeff=0.0,
#                  critic_fit_params=None):
# Algorithm
core = Core(agent, mdp)



# Fill the replay memory with fre_current_two samples
core.learn(n_steps=100, n_steps_per_fit=10)

# RUN


n_steps_test = 100

dataset = core.evaluate(n_steps=n_steps_test, render=False)
J = np.mean(compute_J(dataset))/radarStepNum
print('Epoch: 0')
print('J: ', J )
reward = float('-inf')
rewardRecord=[]
for n in range(n_epochs):
    print('Epoch: ', n+1)
    core.learn(n_steps=n_steps, n_steps_per_fit=10)
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    J = np.mean(compute_J(dataset))/radarStepNum
    print(f'Objective function after learning: {J}')
    reward = max(reward, J)
    rewardRecord.append(J)
save_csv(reward, "attention_PPO", str(jamType) + "_history_two_" + str(rewardType), actionLen,rewardRecord)