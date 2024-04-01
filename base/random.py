from environment.simRadar_fre_current_two import SimRadar
import numpy as np
import random

actionLen = 8
radarStepNum = 10
jamType = 0
radarType = "FMCW"
rewardType = 1
env = SimRadar(actionLen=actionLen, radarStepNum=radarStepNum, jamType=jamType, radarType=radarType,
               rewardType=rewardType)
epsiode=10
epoch=10
totalReward=0
for i in range(epsiode):
    env.reset()
    for i in range(epoch):
        action=random.randint(0,actionLen)
        state,reward,absorbing,_ = env.step(action)
        totalReward+=reward
    print("epsiode:_"+str(epsiode))
    print("reward:_"+str(totalReward))
    print("----------------------------------------")
    totalReward=0

