#!/usr/bin/env python3
"""
Synthetic IQ Test Signal Generator for SDR Demodulator Testing.

Generates artificial AM, NFM, or CW radio signals encoded into complex IQ data
and saves them as raw binary files (.iq) formatted for RTL-SDR readers.
"""

import argparse
import numpy as np


def generate_audio_tone(
    duration_sec: float, sample_rate: int, freq_hz: float = 440.0
) -> np.ndarray:
    """Generates a continuous audio sine wave (e.g., 440 Hz A4 tone)."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Generate 440 Hz tone modulated by a slow 2 Hz pulsing envelope
    pulse_envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 2.0 * t))
    audio = np.sin(2 * np.pi * freq_hz * t) * pulse_envelope
    return audio, t


def synthesize_nfm(
    audio: np.ndarray,
    t: np.ndarray,
    sample_rate: int,
    offset_freq_hz: float = 25000.0,
    max_dev_hz: float = 5000.0,
) -> np.ndarray:
    """
    Synthesizes Narrowband FM (NFM) complex IQ samples.
    
    Phase angle theta[n] is the integral of frequency over time:
    theta[n] = 2 * pi * offset_freq * t + 2 * pi * max_dev * cumsum(audio) / sample_rate
    """
    phase_audio = 2 * np.pi * max_dev_hz * np.cumsum(audio) / sample_rate
    phase_carrier = 2 * np.pi * offset_freq_hz * t
    total_phase = phase_carrier + phase_audio
    
    # Complex exponential carrier wave
    iq_complex = np.exp(1j * total_phase)
    return iq_complex


def synthesize_am(
    audio: np.ndarray,
    t: np.ndarray,
    sample_rate: int,
    offset_freq_hz: float = 25000.0,
    mod_depth: float = 0.8,
) -> np.ndarray:
    """
    Synthesizes Amplitude Modulation (AM) complex IQ samples.
    
    Envelope = 1 + m * audio[t]
    IQ = Envelope * exp(j * 2 * pi * offset_freq * t)
    """
    envelope = 1.0 + mod_depth * audio
    carrier = np.exp(1j * 2 * np.pi * offset_freq_hz * t)
    return envelope * carrier


def add_awgn_noise(iq_complex: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Adds Additive White Gaussian Noise (AWGN) to simulate real atmospheric RF noise."""
    signal_power = np.mean(np.abs(iq_complex) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    
    # Generate complex Gaussian noise (real and imaginary parts)
    noise_i = np.random.normal(0, np.sqrt(noise_power / 2), size=len(iq_complex))
    noise_q = np.random.normal(0, np.sqrt(noise_power / 2), size=len(iq_complex))
    return iq_complex + (noise_i + 1j * noise_q)


def export_uint8_iq(iq_complex: np.ndarray, output_filepath: str):
    """
    Converts normalized complex float samples (-1.0 to +1.0) into 8-bit unsigned
    interleaved IQ bytes (RTL-SDR standard cu8 format: 0 to 255, center 127.5).
    """
    # Extract Real (I) and Imaginary (Q) components
    i_float = np.real(iq_complex)
    q_float = np.imag(iq_complex)
    
    # Scale and clip to uint8 range [0, 255]
    i_uint8 = np.clip(i_float * 127.5 + 127.5, 0, 255).astype(np.uint8)
    q_uint8 = np.clip(q_float * 127.5 + 127.5, 0, 255).astype(np.uint8)
    
    # Interleave I and Q bytes [I0, Q0, I1, Q1, ...]
    interleaved = np.empty((2 * len(iq_complex),), dtype=np.uint8)
    interleaved[0::2] = i_uint8
    interleaved[1::2] = q_uint8
    
    with open(output_filepath, "wb") as f:
        f.write(interleaved.tobytes())

    print(f"[+] Successfully exported {len(iq_complex)} IQ samples to '{output_filepath}'")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic IQ test files.")
    parser.add_argument(
        "--mode", choices=["NFM", "AM"], default="NFM", help="Modulation type"
    )
    parser.add_argument(
        "--output", type=str, default="test_signal.iq", help="Output .iq file path"
    )
    parser.add_argument(
        "--duration", type=float, default=5.0, help="Duration in seconds"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=2400000, help="IQ Sample rate (default: 2.4 MSPS)"
    )
    parser.add_argument(
        "--snr", type=float, default=25.0, help="Signal-to-Noise ratio in dB"
    )
    args = parser.parse_args()

    print(f"[+] Synthesizing {args.duration}s of {args.mode} audio signal...")
    audio, t = generate_audio_tone(duration_sec=args.duration, sample_rate=args.sample_rate)

    if args.mode == "NFM":
        iq_raw = synthesize_nfm(audio, t, sample_rate=args.sample_rate)
    elif args.mode == "AM":
        iq_raw = synthesize_am(audio, t, sample_rate=args.sample_rate)

    # Add realistic RF background noise
    iq_noisy = add_awgn_noise(iq_raw, snr_db=args.snr)

    # Write out raw binary byte stream
    export_uint8_iq(iq_noisy, args.output)


if __name__ == "__main__":
    main()