#!/usr/bin/env python3
"""
CLI entry point for the Guided SDR Demodulator tutorial project.
"""

import argparse
import sys
from io_handlers.iq_source import FileIQSource, LiveRtlSdrSource
from io_handlers.audio_sink import AudioSink
from dsp.demodulators import AMDemodulator, NFMDemodulator, USBDemodulator
from pipeline import SDRPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Production-grade Guided SDR Audio Demodulator in Python."
    )
    parser.add_argument(
        "--mode",
        choices=["NFM", "AM", "USB"],
        default="NFM",
        help="Demodulation scheme (default: NFM)",
    )
    parser.add_argument(
        "--source",
        choices=["live", "file"],
        default="file",
        help="Input source type",
    )
    parser.add_argument(
        "--file", type=str, help="Path to raw binary IQ file (required if --source=file)"
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=145.500e6,
        help="Center frequency in Hz (default: 145.5 MHz)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=2400000,
        help="IQ Sample Rate in Hz (default: 2.4 MSPS)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Select Input IQ Source
    if args.source == "live":
        print(f"[+] Initializing RTL-SDR at {args.freq / 1e6:.3f} MHz...")
        source = LiveRtlSdrSource(
            center_freq_hz=args.freq, sample_rate_hz=args.rate
        )
    else:
        if not args.file:
            print("Error: --file argument is required when using file source.")
            sys.exit(1)
        print(f"[+] Reading IQ Recording: {args.file}")
        source = FileIQSource(filepath=args.file, dtype_str="uint8")

    # 2. Select Demodulator
    if args.mode == "NFM":
        demodulator = NFMDemodulator(tau_us=50.0)
    elif args.mode == "AM":
        demodulator = AMDemodulator()
    elif args.mode == "USB":
        demodulator = USBDemodulator()

    # 3. Initialize Audio Output (Standard 48 kHz PCM)
    audio_sink = AudioSink(sample_rate=48000)

    # 4. Construct & Run Pipeline
    pipeline = SDRPipeline(
        source=source,
        demodulator=demodulator,
        audio_sink=audio_sink,
        input_sample_rate=args.rate,
        target_audio_rate=48000,
    )

    pipeline.run()


if __name__ == "__main__":
    main()