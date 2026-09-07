from typing import Tuple
import numpy as np #not hmmm
import scipy.signal as signal

class LowPassFilter:
    def __init__(self,cutoff_hz:float,sample_rate_hz:float,numtaps:int=101): #btw hz here is the real Hz
        """low pass filter using  hamming window"""
        self.cutoff=cutoff_hz
        self.sample_rate=sample_rate_hz
        self.num_taps=num_taps
        self.taps=signal.firwin(
            num_taps,cutoff_hz,fs=sample_rate_hz,window="hamming"
        )
        self.zi=np.zeros(num_taps-1,dtype=np.complex64)
    def filter(self, samples: np.ndarray) -> np.ndarray:
        """applies the low pass filter to the samples"""
        filtered,self.zi=signal.lfilter(self.taps,1.0,samples,zi=self.zi)
        return filtered

class PolyphaseDecimator:
    def __init__(self,input_rate:int,target_rate:int):
        """
        downsamples high rate iq stream to audio rate and calculates decimation factor"""
        self.decimation_factor=int(input_rate//target_rate)
        self.actual_target_rate=input_rate/self.decimation_factor
        nyquist=self.actual_target_rate /2.0
        self.lpf=LowPassFilter(cutoff_hz=nyquist*0.8,sample_rate_hz=input_rate)

    def process(self,samples:np.ndarray)->Tuple[np.ndarray,float]:
        """filters out high freq noises and downsamples array"""
        filtered=self.lpf.filter(samples)
        #slicing step is the decimation factor
        decimated =filtered[::self.decimation_factor]
        return decimated,self.actual_target_rate

class DeEmphasisFilter:
    def __init__(self,tau_us:float=75.0,sample_rate:float=48000.0):
        """
        low pass filter for fm audio deemphasis"""
        dt=1.0/sample_rate
        tau=tau_us*1e-6
        self.alpha=np.exp(-dt/tau)
        self.last_y=0.0

    def process(self,audio:np.ndarray) ->np.ndarray:
        """applies 1 pole filter to audio samples"""
        out=np.empty_like(audio)
        y=self.last_y
        alpha=self.alpha
        one_minus_alpha=1.0 -alpha

        for i in range(len(audio)):
            y=one_minus_alpha*audio[i]+alpha*y
            out[i]=y
        self.last_y=y
        return out

class AutomaticGainControl:
    def __init__(self,target_level:float=0.5,max_gain:float=100.0):
        """audio agc to maintain balanced volume without clipping"""
        self.target_level=target_level
        self.max_gain=max_gain
        self.gain=1.0
    def process(self,audio:np.ndarray)->np.ndarray:
        peak=np.max(np.abs(audio))
        if peak>1e-6:
            desired_gain=self.target_level/peak
            self.gain=0.95*self.gaim+0.05*min(desired_gain,self.max_gain)
        return np.clip(audio*self.gain,-1.0,1.0)

#feeling bored, so from next file onwards unfortunately im vibecoding