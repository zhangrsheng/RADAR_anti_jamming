import matplotlib.pyplot as plt
import pandas as  pd
import numpy as np
interType="aim"
stateLen="current"
stateForm="two"
# rewardType="sinr"
rewardType="pd"
actionLen=8

interLst=["aim","sweep"]
stateLenLst=["current","history"]
stateFormLst=["single","two"]
rewardLst=["Diff","sinr","pd"]

# x=["A2C","PPO","RANDOM","attentionA2C","attentionPPO"]
x=["A2C","PPO","RANDOM","attentionA2C","attentionPPO"]

df=pd.read_csv(str(interLst.index(interType))+"_"+stateLen+"_"+stateForm+"_"+str(rewardLst.index(rewardType))+".csv")
# y=df.loc[actionLen,:].tolist()
xLst = [actionLen]
attentionA2C=df.loc[xLst,"continous_a2c_attention"].tolist()
attentionPPO=df.loc[xLst,"continous_PPO_attention"].tolist()
A2C=df.loc[xLst,"continous_a2c"].tolist()
PPO=df.loc[xLst,"continous_PPO"].tolist()
RANDOM=df.loc[xLst,"continous_random"].tolist()
y=[A2C[0],PPO[0],RANDOM[0],attentionA2C[0],attentionPPO[0]]


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
plt.savefig("pic//"+ str(interLst.index(interType))+"_"+stateLen+"_"+stateForm+"_"+str(rewardLst.index(rewardType))+"_box_con_"+str(actionLen)+".jpg")
plt.show()

