from environment.continous_radar_history_two import SimRadar
import numpy as np
import random
from fre_current_two.save_res import save_csv
historyLen = 8
# actionLen = 6
# jamType = 0
# rewardType = 2
import sys

actionLen = int(sys.argv[1])
jamType = int(sys.argv[2])
rewardType = int(sys.argv[3])
radarStepNum = 10

radarType = "FMCW"

env = SimRadar(actionLen=actionLen, radarStepNum=radarStepNum, jamType=jamType, radarType=radarType,
               rewardType=rewardType,historyLen=historyLen)
epsiode=10
epoch=300
totalReward=0
rewardMax=float('-inf')
rewardRecord=[]
for i in range(epsiode):
    env.reset()
    for i in range(epoch):
        action = random.uniform(0,actionLen*actionLen)
        state,reward,absorbing,_ = env.step(action)
        totalReward+=reward
    print("epsiode:_"+str(epsiode))
    print("reward:_"+str(totalReward/radarStepNum/epoch))
    print("----------------------------------------")
    rewardMax = max(totalReward/radarStepNum/epoch,rewardMax)
    totalReward=0
    rewardRecord.append(totalReward/radarStepNum/epoch)
save_csv(rewardMax, "continous_random", str(jamType) + "_history_two_" + str(rewardType), actionLen,rewardRecord)
