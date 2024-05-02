from modules import NoiseBandNet
from audio_dataset import AudioDataset
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
  args.add_argument('--audio_chunk_duration', type=float, default=1, help='Duration of audio chunk in seconds')
  args.add_argument('--device', type=str, default='mps', help='Device to use', choices=['cuda', 'cpu', 'mps'])

  config = args.parse_args()

  # Model
  checkpoint = torch.load(config.model_checkpoint, map_location=torch.device(config.device))
  print(f"Checkpoint hyper parameters: {checkpoint['hyper_parameters']}")

  nbn = NoiseBandNet.load_from_checkpoint(config.model_checkpoint)
  # Evaluation mode
  nbn.eval()


  # Dataset
  dataset = AudioDataset(
    dataset_path=config.dataset_path,
    n_signal=int(nbn.samplerate*config.audio_chunk_duration),
    sampling_rate=nbn.samplerate,
  )

  # breakpoint()

  loss = nbn._construct_loss_function()

  output_signal = torch.FloatTensor(0)
  num_samples = config.num_samples if config.num_samples < len(dataset) else len(dataset)
  samples = np.random.choice(range(len(dataset)), num_samples, replace=False)
  print(f"Length of the dataset: {len(dataset)}")
  for i in samples:
    # get example from the dataset
    x_audio = dataset[i].to(nbn.device)

    # pack audio batch
    x_audio = x_audio.unsqueeze(0)

    # forward the control params to generate the audio
    y_audio = nbn(x_audio)

    # calculate the loss
    # loss_val = loss(y_audio, x_audio).item()
    # print(f"Sample {i} Loss: {loss_val}")

    # Unpack audio
    y_audio = y_audio.squeeze(0).detach()

    # concatenate the original and autoencoded audio with the rest of the generation
    silence = torch.zeros(1, int(nbn.samplerate/2))
    output_signal = torch.cat((output_signal, x_audio, silence, y_audio, silence.repeat(1,3)), dim=-1)

  torchaudio.save(config.save_path, output_signal, nbn.samplerate)
