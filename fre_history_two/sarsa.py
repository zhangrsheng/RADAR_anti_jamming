from mushroom_rl.core import Core
from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.dataset import compute_J
import numpy as np
from environment.simRadar_fre_history_two import SimRadar
from fre_history_two.save_res import save_csv

if __name__=="__main__":

    n_steps = 100
    n_steps_per_fit = 10
    n_epochs = 300
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

    env = SimRadar(actionLen=actionLen, radarStepNum=radarStepNum, jamType=jamType, radarType=radarType,
                   rewardType=rewardType, historyLen=historyLen)



    epsilon = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon)
    # print(env.info.observation_space.low)
    # print(env.info.observation_space.high)
    tilings = Tiles.generate(env._actionLen, [0 for _ in range(historyLen*5)],
                             env.info.observation_space.low,
                             env.info.observation_space.high)
    features = Features(tilings=tilings)
    learning_rate = Parameter(.1 / env._actionLen)
    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(env.info.action_space.n,),
                               n_actions=env.info.action_space.n)
    agent = TrueOnlineSARSALambda(env.info, pi,
                                  approximator_params=approximator_params,
                                  features=features,
                                  learning_rate=learning_rate,
                                  lambda_coeff=.9)
    core = Core(agent, env)
    dataset = core.evaluate(n_episodes=1, render=False)
    J = np.mean(compute_J(dataset)) / radarStepNum
    print(f'Objective function before learning: {J}')
    stop = 0
    reward = float('-inf')
    rewardRecord=[]
    for step in range(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=1, render=False)
        dataset = core.evaluate(n_steps=n_steps, render=False)
        # if np.mean(compute_J(dataset, env.info.gamma))<J:
        #     stop+=1
        # if stop >5:
        #     break
        J = np.mean(compute_J(dataset)) / radarStepNum
        print(f'Objective function after learning: {J}')
        reward = max(reward, J)
        rewardRecord.append(J)
    save_csv(reward, "sarsa", str(jamType) + "_history_two_" + str(rewardType), actionLen,rewardRecord)