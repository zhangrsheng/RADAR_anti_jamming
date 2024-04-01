import numpy as np
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box,Discrete
from mushroom_rl.utils.viewer import Viewer
from gym.spaces import MultiDiscrete
from copy import deepcopy
from collections import deque
from radarModel.radarFMCW import RadarFMCW
from radarModel.radarPMCW import RadarPMCW
from jammer.aimed import AimedJam
from jammer.sweep import SweepJam


class SimRadar(Environment):
    def __init__(self,actionLen,radarStepNum,jamType,radarType,rewardType,historyLen):
        self._actionLen = actionLen
        self.radarType=radarType
        self.rewardType=rewardType
        self._aimedJam = AimedJam()
        self._sweepJam = SweepJam(actionLen)
        self._jam = [self._aimedJam, self._sweepJam][jamType]
        self._action_shape = actionLen*actionLen
        self._action_space = Discrete(self._action_shape)


        self._shapeLen=5
        self._historyLen=historyLen

        self._maxLen=historyLen*self._shapeLen
        self._shape = (self._maxLen,)
        self._observation_space=Box(low=0,high=self._actionLen,shape=self._shape)
        mdp_info = MDPInfo(self._observation_space, self._action_space, gamma=0.99, horizon=100)
        super().__init__(mdp_info)
        self._state = None
        self._stateLst = np.zeros([self._historyLen,self._shapeLen])
        self._action=None
        self._radarStepNum=radarStepNum
        self._radarStep=0
        #雷达step数量

    def reset(self, state=None):
        self._stateLst = np.zeros([self._historyLen, self._shapeLen])
        self._radarStep = 0
        self._state = np.zeros(4)
        # if state is None:
        #     self._state=np.fre_current_two.randint(0,self._actionLen,size=2)
        # else:
        #     assert state[0]<self._actionLen
        #     assert state[1]<self._actionLen
        #     self._state=state
        return self._stateLst.flatten()

    def step(self, action):
        # print(action)
        # assert 0 <= action <= self._actionLen
        self._state[2], self._state[3] = self._jam.getJamFre(self._state[0], self._state[1], self._state[2],
                                                             self._state[3])
        # self._state[1] = self._state[0]
        # t时刻jammer频率等于t-1时刻radar频率
        self._state[0] = int(action / self._actionLen)
        self._state[1] = int(action % self._actionLen)

        self._radarState = self._state[0]
        self._radarState = self._state[1]
        # self._state[0],self._state[1] = action
        # print(self._state[0])
        if self.radarType == "FMCW":
            radar = RadarFMCW(self._state[0], self._state[1], self._state[2], self._state[3])
            Diff, snr, pd = radar.runRadar()
            if self.rewardType == 0:
                reward = Diff
            if self.rewardType == 1:
                reward = snr
            if self.rewardType == 2:
                reward = pd
        self._radarStep += 1
        absorbing = self._radarStep >= self._radarStepNum
        # print(self._stateLst)
        # print(action)
        # print(np.array([self._state[0] , self._state[1] , action[0]]))
        self._stateLst = np.insert(self._stateLst, len(self._stateLst), np.array(
            [self._state[0], self._state[1], self._state[2], self._state[3], action[0]]), axis=0)
        self._stateLst = np.array(deque(self._stateLst,
                                        maxlen=self._historyLen
                                        ))
        return self._stateLst.flatten(), reward, absorbing, {}

if __name__ == '__main__':
    mdp = SimRadar(actionLen=8, radarStepNum=10, jamType=0, radarType="FMCW", rewardType=1, historyLen=4)
    print(mdp.reset())