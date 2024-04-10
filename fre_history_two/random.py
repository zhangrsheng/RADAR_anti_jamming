
import numpy as np
import random
from environment.simRadar_fre_history_two import SimRadar
from fre_history_two.save_res import save_csv

n_steps = 1000
n_steps_per_fit = 10
n_epochs = 300
historyLen = 8

actionLen = 6
jamType = 0
rewardType = 2
import sys

if len(sys.argv)>=3:
    actionLen = int(sys.argv[1])
    jamType = int(sys.argv[2])
    rewardType = int(sys.argv[3])
radarStepNum = 10

radarType = "FMCW"

env = SimRadar(actionLen=actionLen, radarStepNum=radarStepNum, jamType=jamType, radarType=radarType,
               rewardType=rewardType, historyLen=historyLen)

epsiode=10
epoch=10
totalReward=0
rewardRecord=[]
rewardMax=float('-inf')
for i in range(epsiode):
    env.reset()
    for i in range(epoch):
        action =np.array([random.randint(0,actionLen*actionLen)])
        print(action)
        state,reward,absorbing,_ = env.step(action)
        totalReward+=reward
    print("epsiode:_"+str(epsiode))
    print("reward:_"+str(totalReward/radarStepNum/epoch))
    print("----------------------------------------")
    rewardMax = max(totalReward/radarStepNum/epoch,rewardMax)
    totalReward=0
    rewardRecord.append(totalReward/radarStepNum/epoch)
save_csv(rewardMax, "random", str(jamType) + "_history_two_" + str(rewardType), actionLen,rewardRecord)
