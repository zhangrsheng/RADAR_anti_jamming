import random

class SweepJam():
    def __init__(self,actionLen):
        self._jam=None
        self._step=0
        self._cycleIndex=0
        self._cycle1=[]
        self._cycle2=[]
        self._cycle3=[]
        for i in range(actionLen-1):
            self._cycle1.append([i,i+2])
        for i in range(actionLen-1):
            self._cycle2.append([actionLen,actionLen-i-1])
        for i in range(actionLen-1):
            self._cycle3.append([i,i+1])
        self._cycle=[self._cycle1,self._cycle2,self._cycle3]
    def reset(self):
        self._step=0
        self._cycleIndex=0
    def getJamFre(self,radarState1,radarState2,jamState1,jamState2):

        if radarState1<=jamState1<=radarState2 or radarState1<=jamState2<=radarState2:
            self._cycleIndex+=0
            self._step+=0
        elif self._step>=len(self._cycle[0])-1:
            self._step=0
            self._cycleIndex = random.randint(0, len(self._cycle) - 1)
        else:
            self._step+=1
        # if self._step>=len(self._cycle[0]):
        #     self._step=0
        #     self._cycleIndex=random.randint(0,len(self._cycle)-1)
        jam=self._cycle[self._cycleIndex][self._step]
        return jam[0],jam[1]