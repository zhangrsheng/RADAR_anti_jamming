import numpy as np
from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.simulator import simc
from radarsimpy.tools import roc_pd, roc_snr
import math

class RadarPMCW():
    def __init__(self,radarF,inetrF):
        self.Diff=abs(radarF-inetrF)
        self.radarF=24e9+radarF*0.05e9
        self.interF=24e9+inetrF*0.05e9
        self.radarMinFacor=0.0001#加在噪声上，防止除以0
        self.pfa=0.00001
        self.pulses=256
    def runRadar(self):
        code1 = np.array([1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1,
                          1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1,
                          1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1,
                          -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1,
                          1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1,
                          1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1,
                          1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1,
                          -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1,
                          -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1,
                          1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1,
                          1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1,
                          -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1,
                          -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1,
                          -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1,
                          1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1,
                          1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1])
        code2 = np.array([1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1,
                          -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1,
                          -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1,
                          -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1,
                          1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1,
                          -1, 1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1,
                          1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1,
                          1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1,
                          -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1,
                          -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1,
                          1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                          1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1,
                          -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1,
                          -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1,
                          1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1,
                          -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1])
        # convert binary code to phases in degrees
        phase_code1 = np.zeros(np.shape(code1))
        phase_code2 = np.zeros(np.shape(code2))
        phase_code1[np.where(code1 == 1)] = 0
        phase_code1[np.where(code1 == -1)] = 180
        phase_code2[np.where(code2 == 1)] = 0
        phase_code2[np.where(code2 == -1)] = 180

        # define modulation timing (4e-9 s per code)
        t_mod1 = np.arange(0, len(phase_code1)) * 4e-9
        t_mod2 = np.arange(0, len(phase_code2)) * 4e-9
        tx_channel_1 = dict(
            location=(0, 0, 0),
            mod_t=t_mod1,
            phs=phase_code1,
        )

        tx_channel_2 = dict(
            location=(1, 0, 0),
            mod_t=t_mod2,
            phs=phase_code2,
        )

        tx = Transmitter(f=self.radarF,
                         t=2.1e-6,
                         tx_power=20,
                         pulses=self.pulses,
                         channels=[tx_channel_1, tx_channel_2])
        rx = Receiver(fs=250e6,
                      noise_figure=10,
                      rf_gain=20,
                      baseband_gain=30,
                      load_resistor=1000,
                      channels=[
                          dict(location=(0, 0, 0))
                      ])
        int_tx = Transmitter(f=self.interF,
                             t=2.1e-6,
                             tx_power=30,
                             pulses=self.pulses,
                             channels=[
                                 dict(location=(0, 0.1, 0))
                             ])

        int_rx = Receiver(fs=250e6,
                          noise_figure=10,
                          rf_gain=20,
                          baseband_gain=30,
                          load_resistor=1000,
                          channels=[
                              dict(location=(0, 0.1, 0))
                          ])
        int_radar = Radar(transmitter=int_tx, receiver=int_rx,
                          location=(30, 0, 0), rotation=(180, 0, 0))

        radar = Radar(transmitter=tx, receiver=rx, interf=int_radar)
        target_1 = dict(location=(30, 0, 0), speed=(0, 0, 0), rcs=10, phase=0)
        target_2 = dict(location=(20, 1, 0), speed=(-10, 0, 0), rcs=10, phase=0)
        targets = [target_1, target_2]
        bb_data = simc(radar, targets, noise=True)
        timestamp = bb_data['timestamp']
        baseband = bb_data['baseband']  # baseband data without interference
        interf = bb_data['interference']
        interf_bb = baseband + interf
        radarAmp=abs(np.mean(abs(baseband)))
        interAmp=abs(np.mean(abs(interf)))+self.radarMinFacor*radarAmp
        snr = 10*math.log10(radarAmp**2/interAmp**2)
        pd = roc_pd(self.pfa, snr, self.pulses, 'Swerling 3')
        return self.Diff, snr, pd

if __name__=="__main__":
    import time

    t1 = time.time()
    radar=RadarPMCW(0,2)
    print(radar.runRadar())
    t2 = time.time()
    print(t2 - t1)