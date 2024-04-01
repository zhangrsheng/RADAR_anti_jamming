import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
interType="aim"        #"aim","sweep"
stateLen="history"     #"current","history"
stateForm="two"     #"single","two"
# rewardType="sinr"
rewardType="sinr"        #"Diff","sinr","pd"
actionLen=8

interLst=["aim","sweep"]
stateLenLst=["current","history"]
stateFormLst=["single","two"]
rewardLst=["Diff","sinr","pd"]

# x=["A2C","DDQN","DQN","PPO","SARSA","RANDOM"]
df=pd.read_csv(str(interLst.index(interType))+"_"+stateLen+"_"+stateForm+"_"+str(rewardLst.index(rewardType))+".csv")
xLst = [4,5,6,7,8,9,10,11,12]
x=[]

attentionA2C=df.loc[xLst,"continous_a2c_attention"].tolist()
attentionPPO=df.loc[xLst,"continous_PPO_attention"].tolist()
A2C=df.loc[xLst,"continous_a2c"].tolist()
PPO=df.loc[xLst,"continous_PPO"].tolist()
# RANDOM=df.loc[xLst,"continous_random"].tolist()
RANDOM=df.loc[xLst,"random"].tolist()





for i in xLst:
    x.append(str(i))


# plt.figure(figsize=(15,8), dpi=100)
plt.figure(figsize = (16,8),dpi=80)
color=["lightcoral","sandybrown","gold","palegreen","turquoise","lightsteelblue","violet","purple"]
plt.plot(x, A2C, c=color[0], label="A2C")
plt.plot(x, PPO, c=color[3], label="PPO")
plt.plot(x, RANDOM, c=color[5], label="RANDOM")
plt.plot(x,attentionA2C,c=color[6],label="attentionA2C")
plt.plot(x,attentionPPO,c=color[7],label="attentionPPO")

plt.scatter(x, A2C, c=color[0])
plt.scatter(x, PPO, c=color[3])
plt.scatter(x, RANDOM, c=color[5])
plt.scatter(x,attentionA2C,c=color[6])
plt.scatter(x,attentionPPO,c=color[7])
plt.legend(loc='best')
if rewardType=="pd":
    plt.yticks(np.arange(0,1.2,0.2))
if rewardType=="sinr":
    plt.yticks(np.arange(0, 90, 10))
if rewardType=="Diff":
    plt.yticks(np.arange(0,actionLen+1,1))

plt.grid(linestyle='--',alpha=0.5)
# plt.grid(True, linestyle='--', alpha=0.8)
plt.xlabel("action space", fontdict={'size': 16})
plt.ylabel(rewardType, fontdict={'size': 16})
plt.title("Algorithm comparison ("+rewardType+")", fontdict={'size': 20})

num1=1.02
num2=0.9
num3=3
num4=0
plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)

plt.savefig("pic//"+ str(interLst.index(interType))+"_"+stateLen+"_"+stateForm+"_"+str(rewardLst.index(rewardType))+"plt_con"+".jpg")
plt.show()
