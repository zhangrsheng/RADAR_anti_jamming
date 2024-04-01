import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core
from mushroom_rl.environments import Atari
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_metrics

from mushroom_rl.utils.parameters import LinearParameter, Parameter
from environment.simRadar_fre_history_two import SimRadar
from fre_history_two.save_res import save_csv

class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

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

    def forward(self, state, action=None):
        features1 = F.relu(self._h1(state.float()))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted

if __name__ == '__main__':
    n_steps = 100
    n_steps_per_fit = 10
    n_epochs = 100
    historyLen = 8

    actionLen = 6
    jamType = 0
    rewardType = 2
    import sys
    actionLen = int(sys.argv[1])
    jamType =int(sys.argv[2])
    rewardType =int( sys.argv[3])
    radarStepNum = 10

    radarType = "FMCW"

    mdp = SimRadar(actionLen=actionLen, radarStepNum=radarStepNum, jamType=jamType, radarType=radarType,
                   rewardType=rewardType, historyLen=historyLen)


    def print_epoch(epoch):
        print('################################################################')
        print('Epoch: ', epoch)
        print('----------------------------------------------------------------')


    def get_stats(dataset):
        score = compute_metrics(dataset)
        print(('min_reward: %f, max_reward: %f, mean_reward: %f,'
               ' median_reward: %f, games_completed: %d' % score))

        return score


    scores = list()

    optimizer = dict()
    optimizer['class'] = optim.Adam
    optimizer['params'] = dict(lr=.00025)

    # Settings
    width = 84
    height = 84
    history_length = 4
    train_frequency = 10
    evaluation_frequency = 25000
    target_update_frequency = 100
    initial_replay_size = 500
    max_replay_size = 5000
    test_samples = 100
    max_steps = 500000

    # Policy
    epsilon = LinearParameter(value=1.,
                              threshold_value=.1,
                              n=1000000)
    epsilon_test = Parameter(value=.05)
    epsilon_random = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon_random)
    input_shape=mdp.info.observation_space.shape

    # input_shape = (2,height,width)
    approximator_params = dict(
        network=Network,
        input_shape=input_shape,
        output_shape=(mdp.info.action_space.n,),
        n_actions=mdp.info.action_space.n,
        n_features=64,
        optimizer=optimizer,
        loss=F.smooth_l1_loss
    )
    approximator = TorchApproximator
    algorithm_params = dict(
        batch_size=32,
        target_update_frequency=target_update_frequency // train_frequency,
        replay_memory=None,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size
    )

    agent = DQN(mdp.info, pi, approximator,
                approximator_params=approximator_params,
                **algorithm_params)

    # Algorithm
    core = Core(agent, mdp)

    print_epoch(0)
    core.learn(n_steps=initial_replay_size,
               n_steps_per_fit=initial_replay_size)

    # Evaluate initial policy
    pi.set_epsilon(epsilon_test)
    dataset = core.evaluate(n_steps=test_samples)
    reward = float('-inf')
    rewardRecord=[]
    for n_epoch in range(0, n_epochs):
        print_epoch(n_epoch)
        print('- Learning:')
        # learning step
        pi.set_epsilon(epsilon)
        core.learn(n_steps=n_steps,
                   n_steps_per_fit=n_steps_per_fit)

        print('- Evaluation:')
        # evaluation step
        pi.set_epsilon(epsilon_test)
        dataset = core.evaluate(n_steps=test_samples)
        # print(dataset)
        J = np.mean(compute_J(dataset))/radarStepNum
        print(f'Objective function after learning: {J}')
        reward = max(reward, J)
        rewardRecord.append(J)
    save_csv(reward, "dqn", str(jamType) + "_history_two_" + str(rewardType), actionLen,rewardRecord)