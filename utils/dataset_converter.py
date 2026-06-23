import os
import argparse
import subprocess
import math
from tqdm import tqdm
from glob import glob

import numpy as np
import soundfile as sf


def normalize_rms(path, target_rms, peak_limit=0.99):
    """Per-track loudness normalize a wav in place: scale so its (channel-mean) RMS == target_rms.

    Training loads wavs AS-IS (no loudness normalization in the data loader), so a multi-style corpus
    MUST be loudness-equalised here, or the loudest tracks dominate the MRSTFT loss and quiet styles
    reconstruct poorly (this is the step that balanced the quiet Harmsworth vs the ~6x louder Morelli).
    If the target gain would clip, it is reduced to keep the peak at `peak_limit`. Returns a note str.
    """
    y, sr = sf.read(path, dtype='float32', always_2d=True)   # [T, C]
    mono = y.mean(axis=1)
    rms = float(np.sqrt((mono ** 2).mean()) + 1e-12)
    gain = target_rms / rms
    peak = float(np.abs(y).max()) * gain
    note = ''
    if peak > peak_limit:                                    # avoid clipping
        gain *= peak_limit / peak
        note = ' [peak-limited]'
    sf.write(path, (y * gain).astype(np.float32), sr)
    return f'rms {rms:.3f}->{target_rms:.2f}{note}'


def convert_to_wav(input_dir, output_dir, sampling_rate, highpass=None, channels=None,
                   normalize_rms_target=None, prefix=''):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_files = glob(os.path.join(input_dir, '**', '*'), recursive=True)
    audio_files = [f for f in audio_files if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.aac', '.aiff'))]

    for audio_file in tqdm(sorted(audio_files)):
        file_name = audio_file.split('/')[-1]
        # `prefix` lets several source datasets be converted into ONE output dir with a stable,
        # grouped sort order (sorted glob order -> territory/style index at training time).
        output_file_path = os.path.join(output_dir, prefix + os.path.splitext(file_name)[0] + '.wav')

        # Format conversion via sox (resample + 16-bit; source channels preserved unless --channels).
        command = ['sox', audio_file, '-r', str(sampling_rate), '-b', '16']
        if channels is not None:
            command.extend(['-c', str(channels)])
        command.append(output_file_path)
        if highpass:
            command.extend(['highpass', str(highpass)])
        subprocess.run(command)

        # Per-track loudness normalization (after format conversion).
        if normalize_rms_target is not None and os.path.exists(output_file_path):
            note = normalize_rms(output_file_path, float(normalize_rms_target))
            tqdm.write(f'  {os.path.basename(output_file_path)[:60]:60s} {note}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert audio -> 16-bit WAV at a sample rate, with '
                                                 'optional per-track RMS loudness normalization.')
    parser.add_argument('--input_dir', type=str, help='Input directory containing audio files')
    parser.add_argument('--output_dir', type=str, help='Output directory for converted audio files')
    parser.add_argument('--sampling_rate', type=int, default=44100, help='Output sample rate')
    parser.add_argument('--highpass', type=int, default=None, help='High-pass filter cutoff (Hz)')
    parser.add_argument('--channels', type=int, default=None, help='Force channel count (default: preserve source)')
    parser.add_argument('--normalize_rms', type=float, default=None,
                        help='Per-track loudness normalize each output to this RMS (e.g. 0.1). '
                             'Required for multi-style datasets so loud tracks do not dominate.')
    parser.add_argument('--prefix', type=str, default='',
                        help='Prefix added to every output filename (combine several sources into one '
                             'output dir with a stable, grouped sort/territory order).')

    args = parser.parse_args()

    convert_to_wav(args.input_dir, args.output_dir, args.sampling_rate, args.highpass, args.channels,
                   normalize_rms_target=args.normalize_rms, prefix=args.prefix)
