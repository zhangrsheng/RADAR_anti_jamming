import os
fileLst=["fre_history_two"]
# fileLst=["fre_current_two","fre_history_two"]
# fileLst=["fre_history_two"]
# alLst1=["DDQN","PPO","attention_PPO","random","continous_random","sarsa"]
alLst1=["a2c","attention_a2c","dqn","DDQN","PPO","attention_PPO","random","continous_random","sarsa"]
alLst2=["continous_a2c","continous_a2c_attention","continous_PPO","continous_ppo_attention"]

alLst=alLst1+alLst2
print(alLst)
# alLst=["continous_a2c","continous_PPO","attention_PPO"]
# fileLst=["fre_history_two"]
# alLst=["a2c","attention_a2c","dqn","DDQN","PPO","random","sarsa"]
actionLenLst=[4,5,6,7,8,9,10,11,12]
jamLst=[0,1]
rewardLst=[1,2]
for file in fileLst:
    for al in alLst:
        for actionLen in actionLenLst:
            for jam in jamLst:
                for reward in rewardLst:
                    print("python "+file+"/"+str(al)+".py "+str(actionLen)+" " +str(jam)+" "+str(reward))
                    os.system("python "+file+"/"+str(al)+".py "+str(actionLen)+" " +str(jam)+" "+str(reward))