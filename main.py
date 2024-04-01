from mushroom_rl.core import Core
from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.utils.dataset import compute_J
from environment.simRadar_fre_current_single import  SimRadar
import numpy as np


environment="SimRadar"
agent="SARSA"
if environment=="SimRadar" and agent=="SARSA":
    epsilon = Parameter(value=1)
    pi = EpsGreedy(epsilon=epsilon)
    env=SimRadar(actionLen=20 , radarStepNum=100)
    print(env.info.observation_space.low)
    print(env.info.observation_space.high)

    tilings = Tiles.generate(env._actionLen,[1,1],
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
    J = np.mean(compute_J(dataset, env.info.gamma))
    print(dataset)
    print(f'Objective function before learning: {J}')
    stop=0
    for step in range(2):
        core.learn(n_steps=20000, n_steps_per_fit=1, render=False)
        dataset = core.evaluate(n_episodes=1, render=False)
        # if np.mean(compute_J(dataset, env.info.gamma))<J:
        #     stop+=1
        # if stop >5:
        #     break
        J=np.mean(compute_J(dataset, env.info.gamma))
    print(f'Objective function after learning: {J}')
