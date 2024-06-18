import os
import argparse
import subprocess
from tqdm import tqdm
from glob import glob

def convert_to_wav(input_dir, output_dir):
    # Check if output directory exists, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all audio files in the input directory
    audio_files = glob(os.path.join(input_dir, '**', '*'), recursive=True)
    audio_files = [f for f in audio_files if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.aac', '.aiff'))]

    for audio_file in tqdm(audio_files):
        file_name = audio_file.split('/')[-1]
        output_file_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.wav')

        # Command to convert audio file to 44100 Hz, mono, 16-bit WAV
        command = [
            'sox',
            audio_file,
            '-r', '44100',             # Sample rate
            '-c', '1',                 # Mono
            '-b', '16',                # 16-bit
            output_file_path
        ]

        # Execute the command using subprocess
        subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert audio files to 44100 Hz, mono, 16-bit WAV format.')
    parser.add_argument('--input_dir', type=str, help='Input directory containing audio files')
    parser.add_argument('--output_dir', type=str, help='Output directory for converted audio files')

    args = parser.parse_args()

    convert_to_wav(args.input_dir, args.output_dir)
