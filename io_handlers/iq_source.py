"""
Abstract Input Source Adapters for reading raw IQ streams.
Supports offline file reading (.iq / .raw) and live RTL-SDR USB dongles.
"""

from abc import ABC, abstractmethod
import numpy as np


class IQSource(ABC):
    @abstractmethod
    def read_chunk(self, num_samples: int) -> np.ndarray:
        """Returns array of complex64 IQ normalized values (-1.0 to +1.0)."""
        pass

    @abstractmethod
    def close(self):
        pass


class FileIQSource(IQSource):
    def __init__(self, filepath: str, dtype_str: str = "uint8"):
        """
        Reads binary IQ files.
        'uint8' / 'cu8': RTL-SDR raw dump format (0 to 255, center 127.5).
        'float32' / 'cf32': Pre-normalized GQRX / SDR# recordings.
        """
        self.file = open(filepath, "rb")
        self.dtype_str = dtype_str

    def read_chunk(self, num_samples: int) -> np.ndarray:
        if self.dtype_str in ["uint8", "cu8"]:
            raw_bytes = self.file.read(num_samples * 2)
            if not raw_bytes:
                return np.array([], dtype=np.complex64)
            data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
            # Normalize offset 127.5 to range [-1.0, 1.0]
            i = (data[0::2] - 127.5) / 127.5
            q = (data[1::2] - 127.5) / 127.5
            return i + 1j * q
        elif self.dtype_str in ["float32", "cf32"]:
            raw_bytes = self.file.read(num_samples * 8)
            if not raw_bytes:
                return np.array([], dtype=np.complex64)
            return np.frombuffer(raw_bytes, dtype=np.complex64)
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype_str}")

    def close(self):
        self.file.close()


class LiveRtlSdrSource(IQSource):
    def __init__(self, center_freq_hz: float, sample_rate_hz: float, gain_db: str = "auto"):
        """Interfaces directly with USB RTL-SDR hardware using librtlsdr."""
        from rtlsdr import RtlSdr

        self.sdr = RtlSdr()
        self.sdr.sample_rate = sample_rate_hz
        self.sdr.center_freq = center_freq_hz
        self.sdr.gain = gain_db

    def read_chunk(self, num_samples: int) -> np.ndarray:
        samples = self.sdr.read_samples(num_samples)
        return samples.astype(np.complex64)

    def close(self):
        self.sdr.close()