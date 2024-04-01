
class AimedJam():
    def __init__(self):
        self._jam=None
    def reset(self):
        self._jam=None
    def getJamFre(self,radarState1,radarState2,jamState1,jamState2):
        return radarState1,radarState2