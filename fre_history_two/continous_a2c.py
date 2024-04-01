import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import A2C
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gym
from mushroom_rl.policy import GaussianTorchPolicy,BoltzmannTorchPolicy
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter
from tqdm import trange
from environment.continous_radar_history_two import SimRadar
from fre_current_two.save_res import save_csv

class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        features1 = torch.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = torch.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment(n_epochs, n_steps, n_steps_per_fit, n_step_test,jamType,actionLen,radarStepNum,radarType,rewardType,historyLen,):
    np.random.seed()

    logger = Logger(A2C.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + A2C.__name__)

    # MDP
    horizon = 1000
    gamma = 0.99
    gamma_eval = 1.
    mdp = SimRadar(actionLen=actionLen, radarStepNum=radarStepNum, jamType=jamType, radarType=radarType,
                   rewardType=rewardType, historyLen=historyLen)
    # Policy
    policy_params = dict(
        n_features=32
    )

    beta = Parameter(1e0)
    # pi = BoltzmannTorchPolicy(Network,
    #                           mdp.info.observation_space.shape,
    #                           mdp.info.action_space.shape,
    #                           beta=beta,
    #                           **policy_params)
    pi = GaussianTorchPolicy(Network,
                                       mdp.info.observation_space.shape,
                                       mdp.info.action_space.shape,
                                       **policy_params)

    # Agent
    critic_params = dict(network=Network,
                         optimizer={'class': optim.RMSprop,
                                    'params': {'lr': 1e-3,
                                               'eps': 1e-5}},
                         loss=F.mse_loss,
                         n_features=32,
                         batch_size=64,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    alg_params = dict(actor_optimizer={'class': optim.RMSprop,
                                       'params': {'lr': 1e-3,
                                                  'eps': 3e-3}},
                      critic_params=critic_params,
                      ent_coeff=0.01
                      )

    agent = A2C(mdp.info, pi, **alg_params)

    # Algorithm
    core = Core(agent, mdp)
    core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)

    # RUN
    dataset = core.evaluate(n_steps=n_step_test, render=False)
    J = np.mean(compute_J(dataset))/radarStepNum
    rewardRecord=[]
    reward=float('-inf')
    for n in range(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_steps=n_step_test, render=False)

        J = np.mean(compute_J(dataset))/radarStepNum
        print(f'Objective function after learning: {J}')
        reward = max(reward, J)
        rewardRecord.append(reward)
    save_csv(reward, "continous_a2c", str(jamType)+"_history_two_" + str(rewardType), actionLen,rewardRecord)


if __name__ == '__main__':

    n_steps = 100
    n_steps_per_fit = 10
    n_epochs = 100

    # actionLen = 6
    # jamType = 0
    # rewardType = 2
    import sys

    actionLen = int(sys.argv[1])
    jamType = int(sys.argv[2])
    rewardType = int(sys.argv[3])
    radarStepNum = 10

    radarType = "FMCW"
    historyLen = 8
    experiment(n_epochs=n_epochs, n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, n_step_test=100,
               actionLen=actionLen,radarStepNum=radarStepNum,jamType=jamType,radarType=radarType,rewardType=rewardType,historyLen=historyLen)
