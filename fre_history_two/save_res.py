import os
import pandas as pd
import numpy as np

# def save_csv(reward,algorithm,name,actionLen,rewardRecord):
#     dataPath=str(name)+".csv"
#     if os.path.exists("..//result//" + dataPath):
#         df=pd.read_csv("..//result//"+dataPath,index_col=False)
#         df.loc[actionLen,algorithm]=reward
#         df.to_csv("..//result//" + dataPath,index=False)
#         print("------------------------save--------------------------")
#     else:
#         df = pd.DataFrame({"a2c":np.zeros(20), "DDQN":np.zeros(20), "dqn":np.zeros(20), "PPO":np.zeros(20), "sarsa":np.zeros(20),"random":np.zeros(20)})
#         df.to_csv("..//result//" + dataPath,index=False)
#         df.loc[actionLen,algorithm]=reward
#         df.to_csv("..//result//" + dataPath,index=False)
#         print("----------------create and save----------------")
#
#     if os.path.exists("..//result//"+str(name)+"_"+str(actionLen)+".csv"):
#         df=pd.read_csv("..//result//"+str(name)+"_"+str(actionLen)+".csv",index_col=False)
#         df[algorithm]=rewardRecord
#         df.to_csv("..//result//"+str(name)+"_"+str(actionLen)+".csv",index=False)
#         print("--------------------save record-----------------------")
#     else:
#         df = pd.DataFrame({"a2c":np.zeros(100), "DDQN":np.zeros(100), "dqn":np.zeros(100), "PPO":np.zeros(100), "sarsa":np.zeros(100),"random":np.zeros(100)})
#         df.to_csv("..//result//"+str(name)+"_"+str(actionLen)+".csv",index=False)
#         df[algorithm] = rewardRecord
#         df.to_csv("..//result//"+str(name)+"_"+str(actionLen)+".csv",index=False)
#         print("--------------create and save  record---------------")
def save_csv(reward,algorithm,name,actionLen,rewardRecord):
    dataPath=str(name)+".csv"
    if os.path.exists("result//" + dataPath):
        df=pd.read_csv("result//"+dataPath,index_col=False)
        df.loc[actionLen,algorithm]=reward
        df.to_csv("result//" + dataPath,index=False)
        print("------------------------save--------------------------")
    else:
        df = pd.DataFrame({"a2c":np.zeros(20), "DDQN":np.zeros(20), "dqn":np.zeros(20), "PPO":np.zeros(20), "sarsa":np.zeros(20),"random":np.zeros(20)})
        df.to_csv("result//" + dataPath,index=False)
        df.loc[actionLen,algorithm]=reward
        df.to_csv("result//" + dataPath,index=False)
        print("----------------create and save----------------")

    if os.path.exists("result//"+str(name)+"_"+str(actionLen)+".csv"):
        df=pd.read_csv("result//"+str(name)+"_"+str(actionLen)+".csv",index_col=False)
        df[algorithm]=rewardRecord
        df.to_csv("result//"+str(name)+"_"+str(actionLen)+".csv",index=False)
        print("--------------------save record-----------------------")
    else:
        df = pd.DataFrame({"a2c":np.zeros(100), "DDQN":np.zeros(100), "dqn":np.zeros(100), "PPO":np.zeros(100), "sarsa":np.zeros(100),"random":np.zeros(100)})
        df.to_csv("result//"+str(name)+"_"+str(actionLen)+".csv",index=False)
        df[algorithm] = rewardRecord
        df.to_csv("result//"+str(name)+"_"+str(actionLen)+".csv",index=False)
        print("--------------create and save  record---------------")
if __name__=="__main__":
    save_csv(5,"PPO","test1",2,[])