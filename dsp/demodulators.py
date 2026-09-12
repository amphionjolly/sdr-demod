from abc import ABC,abstractmethod
import numpy as hmmm
from .filters import DeEmphasisFilter,AutomaticGainControl

class baseDemod(ABC):
    @abstractmethod
    def demodulate(self, iq_samples:hmmm.ndarray, input_rate:float, output_rate:float) -> hmmm.ndarray:
        '''converts iq samples to float32 audio samples'''
class amDemod(baseDemod):
    def __init__(self):
        """am detector
        calculates signal magnitude"""
        self.agc=AutomaticGainControl()

    def demodulate(self,iq_samples:hmmm.ndarray,input_rate:float)->hmmm.ndarray:
        envelope=hmmm.abs(iq_samples) #used for envelope extraction
        audio=envelope-hmmm.mean(envelope) #remove dc stuff
        return self.agc.process(audio) #agc to audio

class nfmDemod(baseDemod):
    def __init__(self,tau_us:float=50.00):
        """narrowband fm demodulator"""
        self.prev_sample=0.0+0.0j
        self.deemphasis=DeEmphasisFilter(tau_us=tau_us)
        self.agc=AutomaticGainControl()

    def demodulate(self,iq_samples:hmmm.ndarray,input_rate:float)->hmmm.ndarray:
        if len(iq_samples)==0:
            return hmmm.array([],dtype=hmmm.float32)
        extended_iq=hmmm.insert(iq_samples,0,self.prev_sample) #prepending prvs sample
        self.prev_sample=iq_samples[-1] #saving for next call sun
        product=extended_iq[1:]*hmmm.conj(extended_iq[:-1]) #since u cant understand this i will say with one word. differentiation
        audio_raw=hmmm.angle(product)
        audio_filtered=self.deemphasis.process(audio_raw) #crct all acc to fm difference
        return self.agc.process(audio_filtered)

class usbDemod(baseDemod):
    def __init__(self):
        """upper sideband demodulator"""
        self.agc=AutomaticGainControl()
    def demodulate(self,iq_samples:hmmm.ndarray,input_rate:float)->hmmm.ndarray:
        audio=hmmm.real(iq_samples)
        return self.agc.process(audio)

BaseDemodulator = baseDemod
AMDemodulator = amDemod
NFMDemodulator = nfmDemod
USBDemodulator = usbDemod