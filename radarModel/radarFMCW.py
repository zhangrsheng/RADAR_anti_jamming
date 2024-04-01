import numpy as np
from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.simulator import simc
from radarsimpy.tools import roc_pd, roc_snr
import math
from radarModel.radar import RadarModel

class RadarFMCW():
    def __init__(self,radarF1,radarF2,interF1,interF2):
        self.Diff=abs((radarF1+radarF2)/2-(interF1-interF2)/2)
        self.radarF1=60e9+radarF1*0.05e9
        self.radarF2=60e9+radarF2*0.05e9
        self.interF1=60e9+interF1*0.05e9
        self.interF2=60e9+interF2*0.05e9
        self.radarMinFacor = 0.0001  # 加在噪声上，防止除以0
        self.pulses=4
        self.pfa=0.00001

    def runRadar(self):
        f_offset_int = np.arange(0, 4) * 70e6
        int_tx = Transmitter(f=[self.interF1, self.interF2],
                             t=[0, 8e-6],
                             tx_power=30,
                             prp=20e-6,
                             pulses=self.pulses,
                             f_offset=f_offset_int,
                             channels=[dict(location=(0, 0.1, 0),
                                            pulse_phs=np.array([180, 0, 0, 0]))])

        int_rx = Receiver(fs=20e6,
                          noise_figure=8,
                          rf_gain=20,
                          load_resistor=500,
                          baseband_gain=30,
                          channels=[dict(location=(0, 0.1, 0))])
        int_radar = Radar(transmitter=int_tx, receiver=int_rx,
                          location=(30, 0, 0), rotation=(180, 0, 0))

        f_offset_vit = np.arange(0, 4) * 90e6
        tx = Transmitter(f=[self.radarF1, self.radarF2],
                         t=[0, 8e-6],
                         tx_power=20,
                         prp=20e-6,
                         pulses=self.pulses,
                         f_offset=f_offset_vit,
                         channels=[dict(location=(0, 0, 0),
                                        pulse_phs=np.array([180, 0, 0, 0]))])

        rx = Receiver(fs=40e6,
                      noise_figure=2,
                      rf_gain=20,
                      load_resistor=500,
                      baseband_gain=60,
                      channels=[dict(location=(0, 0, 0))])

        radar = Radar(transmitter=tx, receiver=rx, interf=int_radar)
        target_1 = dict(location=(30, 0, 0), speed=(0, 0, 0), rcs=10, phase=0)
        target_2 = dict(location=(20, 1, 0), speed=(-10, 0, 0), rcs=10, phase=0)

        targets = [target_1, target_2]
        bb_data = simc(radar, targets, noise=False)
        timestamp = bb_data['timestamp']
        baseband = bb_data['baseband']  # baseband data without interference
        interf = bb_data['interference']  # interference data
        interf_bb = baseband + interf  # baseband data with interference

        radarAmp=abs(np.mean(abs(baseband)))
        interAmp=abs(np.mean(abs(interf)))+self.radarMinFacor*radarAmp
        snr = 10 * math.log10(radarAmp ** 2 / interAmp ** 2)
        pd = roc_pd(self.pfa, snr, self.pulses, 'Swerling 3')
        return self.Diff, snr, pd

if __name__ == "__main__":
    import time

    t1 = time.time()
    radar = RadarFMCW(0, 2,4,5)
    print(radar.runRadar())
    t2 = time.time()
    print(t2 - t1)