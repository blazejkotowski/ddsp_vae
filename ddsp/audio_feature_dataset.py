import torch
from torch.utils.data import Dataset
import librosa as li
import os
from glob import glob
import lmdb
import pickle
import hashlib
import numpy as np
import math

from ddsp.interfaces import ControlSpace
from ddsp.registry import FEATURE_EXTRACTORS

from typing import Callable, List


class AudioFeatureDataset(Dataset):
  def __init__(self,
               dataset_path: str,
               n_signal: int,
               sampling_rate: int,
               resampling_factor: int,
               control_space: ControlSpace,
               transform_fn: Callable = None,
               n_channels: int = 1,
               device: str = 'cuda'):
    """
    Arguments:
      - dataset_path: str, the path to the dataset
      - n_signal: int, the size of the audio chunks, in samples
      - sampling_rate: int, the sampling rate of the audio
      - n_channels: int, the number of audio channels (files are zero-padded/cropped to this count)
      - device: str, the device to use
    """
    self._device = device
    self._dataset_path = dataset_path
    self._n_signal = n_signal
    self._sampling_rate = sampling_rate
    self._resampling_factor = resampling_factor
    self._transform_fn = transform_fn
    self._control_space = control_space
    self._n_channels = n_channels
    # Per-file sample lengths of the concatenated audio (for boundary-respecting
    # windowing in the prior token cache). Populated on load; persisted in the cache.
    self._file_lengths = None

    # Build feature extractors from control space (feature fields only)
    self._extractor_specs: List[tuple[str, object, dict]] = []
    for field in self._control_space.fields:
      if field.source == 'feature':
        if not field.extractor:
          raise ValueError(f"ControlField '{field.name}' requires 'extractor' for source='feature'")
        extractor = FEATURE_EXTRACTORS.create(field.extractor, **(field.params or {}))
        norm = dict(field.normalization) if field.normalization is not None else {}
        self._extractor_specs.append((field.name, extractor, norm))

    # Create cache key based on dataset parameters
    cache_key = self._create_cache_key(dataset_path, n_signal, sampling_rate, resampling_factor, control_space)
    self._db_path = os.path.join(os.path.dirname(dataset_path), f"audio_cache_{cache_key}.lmdb")

    # Try to load from LMDB first
    if os.path.exists(self._db_path):
      print(f"Loading from cache: {self._db_path}")
      self._load_from_cache()
    else:
      print(f"Creating new cache: {self._db_path}")
      self._create_cache(dataset_path)

  def _create_cache_key(self, dataset_path: str, n_signal: int, sampling_rate: int, resampling_factor: int, control_space: ControlSpace) -> str:
    """Create a unique cache key based on dataset parameters and control space."""
    field_sig = ";".join([f"{f.name}:{f.source}:{f.dim}:{f.extractor}:{sorted((f.params or {}).items())}" for f in control_space.fields])
    key_string = f"{dataset_path}_{n_signal}_{sampling_rate}_{resampling_factor}_{self._n_channels}ch_{field_sig}"
    return hashlib.md5(key_string.encode()).hexdigest()[:8]

  def _create_cache(self, dataset_path: str):
    """Load audio, process it, and save to LMDB."""
    # Load and process audio -> [n_channels, T_total]
    self._audio = self._load_dataset(dataset_path).to(self._device)
    self._dataset_length = int(self._audio.shape[-1] // self._n_signal)

    # Extract features
    self._features = self._extract_features()

    # Save to LMDB
    self._save_to_cache()


  def _iter_chunks(self, N: int, chunk_samps: int):
    for i in range(0, N, chunk_samps):
        j = min(i + chunk_samps, N)
        yield i, j


  def _put_with_resize(self, env, put_fn):
    """Run a write block; auto-grow map_size on MapFullError."""
    while True:
      try:
        with env.begin(write=True) as txn:
          put_fn(txn)
        break
      except lmdb.MapFullError:
        info = env.info()
        env.set_mapsize(int(info["map_size"] * 1.5))


  def _save_to_cache(self):
    """
    Save audio/features to LMDB in chunks to stay under LMDB's ~2 GiB per-value limit.
    """
    chunk_samps = 10_000_000
    N = int(self._audio.shape[-1])
    F = int(self._features.shape[-1])

    env = lmdb.open(
      self._db_path,
      map_size=16 * 1024**3,  # 16 GiB to start
      readahead=False,
      writemap=False,
      max_dbs=1,
      sync=True,
      metasync=True,
      subdir=True,
    )

    # write metadata
    def put_meta(txn):
      metadata = {
        "dataset_length": self._dataset_length,
        "n_signal": self._n_signal,
        "sampling_rate": self._sampling_rate,
        "resampling_factor": self._resampling_factor,
        "n_channels": int(self._n_channels),
        "total_samps": N,
        "feat_dim": F,
        "chunk_samps": chunk_samps,
        "dtype": "float32",
        "file_lengths": self._file_lengths,
      }
      print(f"Saving metadata: {metadata}")
      txn.put(b"metadata", pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL))

    self._put_with_resize(env, put_meta)

    bytes_in_txn = 0
    txn = env.begin(write=True)
    try:
      for idx, (a, b) in enumerate(self._iter_chunks(N, chunk_samps)):
        aud = self._audio[:, a:b].detach().to("cpu").contiguous().to(torch.float32)
        fea = self._features[a:b].detach().to("cpu").contiguous().to(torch.float32)

        aud_bytes = aud.numpy().tobytes(order="C")
        fea_bytes = fea.numpy().tobytes(order="C")

        txn.put(f"audio:{idx:08d}".encode(), aud_bytes)
        txn.put(f"feat:{idx:08d}".encode(), fea_bytes)

        bytes_in_txn += len(aud_bytes) + len(fea_bytes)

        if bytes_in_txn >= 256 * 1024**2:  # commit every ~256 MiB
          txn.commit()
          txn = env.begin(write=True)
          bytes_in_txn = 0

      txn.commit()
    finally:
      env.close()


  def _load_from_cache(self):
    """Load audio/features from LMDB chunks and reassemble."""
    env = lmdb.open(self._db_path, readonly=True, lock=False, readahead=True, subdir=True)

    with env.begin() as txn:
      meta = pickle.loads(txn.get(b"metadata"))
      self._dataset_length = int(meta["dataset_length"])
      N = int(meta.get("total_samps", 0)) or (self._dataset_length * self._n_signal)
      F = int(meta.get("feat_dim", 0))
      n_channels = int(meta.get("n_channels", 1))
      cs = int(meta["chunk_samps"])
      dtype = np.float32
      self._file_lengths = meta.get("file_lengths", None)

      n_chunks = math.ceil(N / cs)
      audio_cpu = torch.empty((n_channels, N), dtype=torch.float32)
      feats_cpu = torch.empty((N, F), dtype=torch.float32)

      pos = 0
      for idx in range(n_chunks):
        a_key = f"audio:{idx:08d}".encode()
        f_key = f"feat:{idx:08d}".encode()

        a_buf = txn.get(a_key)
        f_buf = txn.get(f_key)
        if a_buf is None or f_buf is None:
          raise RuntimeError(f"Missing chunk {idx} in LMDB")

        a_arr = np.frombuffer(a_buf, dtype=dtype)
        n_samps = a_arr.shape[0] // n_channels
        a_arr = a_arr.reshape(n_channels, n_samps)
        f_arr = np.frombuffer(f_buf, dtype=dtype).reshape(n_samps, F)

        audio_cpu[:, pos:pos+n_samps] = torch.from_numpy(a_arr)
        feats_cpu[pos:pos+n_samps] = torch.from_numpy(f_arr)
        pos += n_samps

    env.close()
    self._audio = audio_cpu.to(self._device, non_blocking=True)
    self._features = feats_cpu.to(self._device, non_blocking=True)


  def __len__(self):
    return self._dataset_length


  def get_datapoint(self, idx):
    sample_start = int(idx * self._n_signal)
    sample_end = int(sample_start + self._n_signal)

    total_samps = self._audio.shape[-1]
    if sample_end > total_samps:
      audio = torch.cat([self._audio[:, sample_start:], self._audio[:, :sample_end - total_samps]], dim=-1)
      features = torch.cat([self._features[sample_start:], self._features[:sample_end - len(self._features)]])

    else:
      audio = self._audio[:, sample_start:sample_end]
      features = self._features[sample_start:sample_end]

    if self._transform_fn is not None:
      audio = self._transform_fn(audio)

    return audio, features


  def __getitem__(self, idx):
    idx = idx % self._dataset_length

    audio, features = self.get_datapoint(idx)

    # Downsample features once to control sequence length
    # features: [T_audio, D_feat]
    T_audio = features.shape[0]
    T_ctl = math.ceil(T_audio / self._resampling_factor)
    # interpolate along time to T_ctl using torch
    feat_ds = torch.nn.functional.interpolate(
      features.T.unsqueeze(0),  # [1, D_feat, T_audio]
      size=T_ctl,
      mode='linear',
      align_corners=False,
    ).squeeze(0).T  # [T_ctl, D_feat]

    return audio, feat_ds


  def _load_dataset(self, path: str) -> torch.Tensor:
    """
    Load and concat entire dataset into single tensor.

    Arguments:
      - path: str, path to the dataset
    Returns:
      - audio: torch.Tensor, the audio tensor
    """
    if not os.path.exists(path):
      raise FileNotFoundError(f"Dataset path does not exist: {path}")

    filepaths = glob(os.path.join(path, '**', '*.wav'), recursive=True)
    if len(filepaths) == 0:
      raise RuntimeError(f"No audio files found in path: {path}")

    audio = torch.zeros((self._n_channels, 0), device=self._device)
    file_lengths = []
    for filepath in filepaths:
      x = self._load_file(filepath)  # [n_channels, T]
      file_lengths.append(int(x.shape[-1]))
      audio = torch.concat([audio, torch.from_numpy(x).to(self._device)], dim=-1)
    self._file_lengths = file_lengths
    return audio

  def file_lengths(self):
    """Per-file sample lengths of the concatenated audio (sums to total samples).
    Falls back to reconstructing from file headers (for older audio caches that
    predate this metadata), rescaled to the dataset sampling rate and adjusted so
    the lengths sum exactly to the loaded total."""
    if self._file_lengths is not None:
      return self._file_lengths
    try:
      import soundfile as sf
      filepaths = glob(os.path.join(self._dataset_path, '**', '*.wav'), recursive=True)
      if len(filepaths) == 0:
        return None
      lens = []
      for fp in filepaths:
        info = sf.info(fp)
        lens.append(int(round(info.frames * float(self._sampling_rate) / float(info.samplerate))))
      total = int(self._audio.shape[-1])
      drift = total - sum(lens)
      if lens:
        lens[-1] += drift  # absorb rounding drift into the last file
      self._file_lengths = lens
      return lens
    except Exception:
      return None


  def _load_file(self, path: str):
    """
    Load an audio file from a path and conform it to n_channels.
    Returns a numpy array of shape [n_channels, n_samples].
    """
    audio, _ = li.load(path, sr = self._sampling_rate, mono = False)
    # librosa returns [n_samples] for mono and [n_channels, n_samples] otherwise
    audio = np.atleast_2d(audio)

    n_channels = audio.shape[0]
    if n_channels > self._n_channels:
      audio = audio[:self._n_channels]
    elif n_channels < self._n_channels:
      padding = np.zeros((self._n_channels - n_channels, audio.shape[-1]), dtype=audio.dtype)
      audio = np.concatenate([audio, padding], axis=0)
    return audio


  def _extract_features(self):
    """
    Extract features from the audio dataset.
    This is a placeholder function, as the actual feature extraction logic is not provided.
    """
    # Compute audio-rate features from a mono downmix of the full concatenated audio.
    # Sum over channels so a single shared feature/control stream drives all channels.
    mono = self._audio.sum(dim=0)  # [T_total]
    feats = []
    for name, extractor, norm in self._extractor_specs:
      feat = extractor(mono, self._sampling_rate)  # [N, C] at audio rate
      if feat.ndim == 1:
        feat = feat.unsqueeze(-1)
      # Apply optional normalization per field
      if 'mean' in norm and 'std' in norm:
        mean = torch.as_tensor(norm['mean'], device=feat.device, dtype=feat.dtype)
        std = torch.as_tensor(norm['std'], device=feat.device, dtype=feat.dtype)
        feat = (feat - mean) / (std + 1e-8)
      elif 'min' in norm and 'max' in norm:
        minv = torch.as_tensor(norm['min'], device=feat.device, dtype=feat.dtype)
        maxv = torch.as_tensor(norm['max'], device=feat.device, dtype=feat.dtype)
        feat = (feat - minv) / (maxv - minv + 1e-8)
      feats.append(feat)

    features = torch.cat(feats, dim=-1)  # [N, D_feat]
    return features
