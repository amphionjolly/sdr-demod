"""
Real-time soundcard playback engine using sounddevice/PortAudio.
"""

import sounddevice as sd
import numpy as np


class AudioSink:
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        # Open non-blocking PCM output stream
        self.stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32"
        )
        self.stream.start()

    def play(self, audio_chunk: np.ndarray):
        """Pushes float32 audio block directly to system speakers."""
        if len(audio_chunk) > 0:
            # Ensure 2D column shape expected by PortAudio
            formatted = np.ascontiguousarray(audio_chunk, dtype=np.float32).reshape(-1, 1)
            self.stream.write(formatted)

    def stop(self):
        self.stream.stop()
        self.stream.close()