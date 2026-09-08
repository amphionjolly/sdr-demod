"""
Main SDR Processing Pipeline. Orchestrates IQ fetching, decimation,
demodulation, and audio playback in a low-latency loop.
"""

import time
import numpy as np
from io_handlers.iq_source import IQSource
from io_handlers.audio_sink import AudioSink
from dsp.filters import PolyphaseDecimator
from dsp.demodulators import BaseDemodulator


class SDRPipeline:
    def __init__(
        self,
        source: IQSource,
        demodulator: BaseDemodulator,
        audio_sink: AudioSink,
        input_sample_rate: int = 2400000,
        target_audio_rate: int = 48000,
        chunk_size: int = 262144,
    ):
        self.source = source
        self.demodulator = demodulator
        self.sink = audio_sink
        self.chunk_size = chunk_size
        self.is_running = False

        # Initialize multi-stage polyphase decimator
        self.decimator = PolyphaseDecimator(
            input_rate=input_sample_rate, target_rate=target_audio_rate
        )

    def run(self):
        """Executes real-time DSP loop."""
        self.is_running = True
        print("[+] SDR Processing Pipeline Active. Press Ctrl+C to stop.")

        try:
            while self.is_running:
                # 1. Fetch raw IQ block
                raw_iq = self.source.read_chunk(self.chunk_size)
                if len(raw_iq) == 0:
                    print("[!] End of IQ stream or file reached.")
                    break

                # 2. Downsample and filter baseband
                audio_rate_iq, actual_rate = self.decimator.process(raw_iq)

                # 3. Demodulate complex samples to audio float
                audio_pcm = self.demodulator.demodulate(audio_rate_iq, actual_rate)

                # 4. Output to soundcard
                self.sink.play(audio_pcm)

        except KeyboardInterrupt:
            print("\n[-] Pipeline stopped by user.")
        finally:
            self.cleanup()

    def cleanup(self):
        self.is_running = False
        self.source.close()
        self.sink.stop()
        print("[+] Hardware and audio streams released gracefully.")