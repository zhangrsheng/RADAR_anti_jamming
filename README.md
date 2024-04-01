environment：两个环境。continuous是连续动作空间的环境，另一个是离散空间。

fre_history_two:里面是全部的算法，带continuous的是连续动作空间，带attention的是加入了注意力机制

jammer：两种干扰，一个是扫频（sweep），在环境中jamType=1。一个是瞄频（aimed），在环境中jamType=0。

radarModel：雷达模型，radarFMCW.py。三个奖励方式，一个是频率差值，在环境中rewardType=0。一个是信干比，在环境中rewardType=1。一个是检测概率，在环境中rewardType=2。

radarsimpy：支持雷达模型的库文件。

result：运行后生成的csv文件存储地。包含五个画图文件。

run：所有算法批量运行

使用方法：fre_history_two中每个算法都能单独运行，命令行参数为actionLen（动作空间大小，目前设置在4-12）jamType（干扰类型，数字，见jammer）rewardType（奖励类型，数字，见radarModel）
