import matplotlib.pyplot as plt
import pandas as  pd
import numpy as np
interType="aim"
stateLen="current"
stateForm="two"
# rewardType="sinr"
rewardType="sinr"
actionLen=8

interLst=["aim","sweep"]
stateLenLst=["current","history"]
stateFormLst=["single","two"]
rewardLst=["Diff","sinr","pd"]

# x=["A2C","DDQN","DQN","PPO","SARSA","RANDOM","attentionA2C","attentionPPO"]
x=["A2C","DDQN","DQN","SARSA","RANDOM","attentionA2C"]

df=pd.read_csv(str(interLst.index(interType))+"_"+stateLen+"_"+stateForm+"_"+str(rewardLst.index(rewardType))+".csv")
# y=df.loc[actionLen,:].tolist()
xLst = [actionLen]
A2C=df.loc[xLst,"a2c"].tolist()
# attentionA2C=df.loc[xLst,"attention_a2c"].tolist()
attentionPPO=df.loc[xLst,"attention_PPO"].tolist()
DDQN=df.loc[xLst,"DDQN"].tolist()
DQN=df.loc[xLst,"dqn"].tolist()
# PPO=df.loc[xLst,"PPO"].tolist()
SARSA=df.loc[xLst,"sarsa"].tolist()
RANDOM=df.loc[xLst,"random"].tolist()
y=[A2C[0],DDQN[0],DQN[0],SARSA[0],RANDOM[0],attentionPPO[0]]


plt.figure(figsize = (16,8),dpi=80)
#3、绘制柱状图
x_ticks = range(len(x))
plt.bar(x_ticks,y,width=0.3,label=rewardType)
plt.legend()
#4、修改X刻度
plt.xticks(x_ticks,x)
#添加网格显示
plt.grid(linestyle='--',alpha=0.5)
plt.title("Algorithm comparison ("+rewardType+")")
#5、标题
plt.savefig("pic//"+ str(interLst.index(interType))+"_"+stateLen+"_"+stateForm+"_"+str(rewardLst.index(rewardType))+"_box_dis_"+str(actionLen)+".pdf")
plt.show()


