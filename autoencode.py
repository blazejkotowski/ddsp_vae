from modules import NoiseBandNet
from lib.dataset_tool import AudioDataset
import argparse
import torch
import torchaudio
import torch.nn.functional as F

if __name__ == '__main__':
  args = argparse.ArgumentParser()
  args.add_argument('--model_checkpoint', type=str, help='Path to the model checkpoint')
  args.add_argument('--dataset_path', type=str, help='Directory of the training sound/sounds')
  args.add_argument('--save_path', type=str, help='Directory to save the autoencoded audio')
  args.add_argument('--num_samples', type=int, default=10, help='Number of samples to autoencode')
  args.add_argument('--resampling_factor', type=int, default=32, help='Resampling factor for the control signal and noise bands')
  args.add_argument('--audio_chunk_duration', type=float, default=1, help='Duration of audio chunk in seconds')
  args.add_argument('--fs', type=int, default=44100, help='Sampling rate of the audio')

  config = args.parse_args()

  dataset = AudioDataset(
    dataset_path=config.dataset_path,
    audio_size_samples=int(config.fs*config.audio_chunk_duration),
    min_batch_size=1,
    sampling_rate=config.fs,
    device='cpu'
  )

  nbn = NoiseBandNet.load_from_checkpoint(config.model_checkpoint)
  # Evaluation mode
  nbn.eval()

  output_signal = torch.FloatTensor(0)
  for i in range(config.num_samples):
    # get example from the dataset
    x_audio, control_params = dataset[i]

    # reshape and downsample the control params
    control_params = [F.interpolate(c.unsqueeze(0), scale_factor = 1/config.resampling_factor, mode='linear') for c in control_params]

    # forward the control params to generate the audio
    y_audio = nbn(control_params).squeeze(0).detach()

    # concatenate the original and autoencoded audio with the rest of the generation
    silence = torch.zeros(1, int(config.fs/2))
    output_signal = torch.cat((output_signal, x_audio, silence, y_audio, silence.repeat(1,3)), dim=-1)

  torchaudio.save(config.save_path, output_signal, config.fs)
