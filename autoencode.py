from modules import NoiseBandNet
from lib.dataset_tool import AudioDataset
import argparse
import torch
import torchaudio
import numpy as np
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

  # Model
  checkpoint = torch.load(config.model_checkpoint, map_location=torch.device('mps'))
  print(f"Checkpoint hyper parameters: {checkpoint['hyper_parameters']}")

  # Ybalferran
  # nbn = NoiseBandNet.load_from_checkpoint(config.model_checkpoint,
  #                                         n_control_params=3,
  #                                         torch_device='mps') # only for ybalferran model
  #                                         # n_control_params=2) # only for footsteps model

  # Ybalferran-Lat512
  nbn = NoiseBandNet.load_from_checkpoint(config.model_checkpoint)
  # Evaluation mode
  nbn.eval()


  # Dataset
  dataset = AudioDataset(
    dataset_path=config.dataset_path,
    audio_size_samples=int(config.fs*config.audio_chunk_duration),
    min_batch_size=1,
    sampling_rate=config.fs,
    auto_control_params=['loudness', 'centroid', 'flatness'], # for ybalferran model
    # auto_control_params=['loudness', 'centroid'], # for footsteps and ybalferran-512 model
    device='cpu'
  )

  # breakpoint()

  loss = nbn._construct_loss_function()

  output_signal = torch.FloatTensor(0)
  num_samples = config.num_samples if config.num_samples < len(dataset) else len(dataset)
  samples = np.random.choice(range(len(dataset)), num_samples, replace=False)
  print(f"Length of the dataset: {len(dataset)}")
  for i in samples:
    # get example from the dataset
    x_audio, control_params = dataset[i]

    # pack audio batch
    x_audio = x_audio.unsqueeze(0)

    # reshape and downsample the control params
    control_params = [F.interpolate(c.unsqueeze(0), scale_factor = 1/config.resampling_factor, mode='linear') for c in control_params]

    # forward the control params to generate the audio
    y_audio = nbn(control_params)

    # calcualte the loss
    # loss_val = loss(y_audio, x_audio).item()
    # print(f"Sample {i} Loss: {loss_val}")

    # Unpack audio
    y_audio = y_audio.squeeze(0).detach()

    # concatenate the original and autoencoded audio with the rest of the generation
    silence = torch.zeros(1, int(config.fs/2))
    output_signal = torch.cat((output_signal, x_audio.squeeze(0), silence, y_audio, silence.repeat(1,3)), dim=-1)

  torchaudio.save(config.save_path, output_signal, config.fs)
