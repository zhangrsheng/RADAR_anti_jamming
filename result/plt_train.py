import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
interType="aim"        #"aim","sweep"
stateLen="history"     #"current","history"
stateForm="two"     #"single","two"
# rewardType="sinr"
rewardType="sinr"        #"Diff","sinr","pd"
al="A2C"  #"attention_A2C"
colName1="a2c"
colName2="attention_a2c"
actionLen=4


interLst=["aim","sweep"]
stateLenLst=["current","history"]
stateFormLst=["single","two"]
rewardLst=["Diff","sinr","pd"]

# x=["A2C","DDQN","DQN","PPO","SARSA","RANDOM"]
df=pd.read_csv(str(interLst.index(interType))+"_"+stateLen+"_"+stateForm+"_"+str(rewardLst.index(rewardType))+"_"+str(actionLen)+".csv")
# xLst = [4,5,6,7,8,9,10,11,12]
x=range(300)
A2C=df.loc[:,colName1].tolist()
attention_A2C=df.loc[:,colName2].tolist()
# for i in range(len(A2C)):
#         if i>120:
#             A2C[i]*=1.8
# attentionA2C=df.loc[xLst,"attention_a2c"].tolist()
# DDQN=df.loc[xLst,"DDQN"].tolist()
# DQN=df.loc[xLst,"dqn"].tolist()
# PPO=df.loc[xLst,"PPO"].tolist()
# SARSA=df.loc[xLst,"sarsa"].tolist()
# RANDOM=df.loc[xLst,"random"].tolist()

# for i in xLst:
#     x.append(str(i))


# plt.figure(figsize=(15,8), dpi=100)
plt.figure(figsize = (16,8),dpi=100)
color=["lightcoral","sandybrown","gold","palegreen","turquoise","lightsteelblue","violet","orange"]
plt.plot(x, A2C, c=color[0], label="A2C")
plt.plot(x,attention_A2C,c=color[4],label="attention-A2C")
plt.legend()


# plt.plot(x, DDQN, c=color[1], label="DDQN")
# plt.plot(x, DQN, c=color[2], label="DQN")
# plt.plot(x, PPO, c=color[3], label="PPO")
# plt.plot(x, SARSA, c=color[4], label="SARSA")
# plt.plot(x, RANDOM, c=color[5], label="RANDOM")
# plt.plot(x,attentionA2C,c=color[6],label="attentionA2C")
# plt.plot(x,attentionA2C,c=color[6],label="attentionA2C")

# plt.legend(loc='best')
# plt.yticks(np.arange(0,1.2,0.2))
plt.grid(linestyle='--',alpha=0.5)
# plt.grid(True, linestyle='--', alpha=0.8)
plt.xlabel("epoch", fontdict={'size': 20})
plt.ylabel(rewardType, fontdict={'size': 20})
# plt.title("train ("+rewardType+","+al+")", fontdict={'size': 20})
plt.title("train" , fontdict={'size': 20})

num1=1.02
num2=0.9
num3=3
num4=0
# plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)

plt.savefig("pic//"+ str(interLst.index(interType))+"_"+stateLen+"_"+stateForm+"_"+str(rewardLst.index(rewardType))+"_"+al+"_"+str(actionLen)+".pdf")
plt.show()
