Example usage:

## Examples

### VAE-based ddsp model (with latent space only)
```python
builder = DDSPBuilder()

# Define the network
builder.add_encoder('vae', latent_size=8, beta=0.1)
builder.add_synth_block('noiseband', n_filters=512)
builder.add_synth_block('harmonic', n_harmonics=500)
ddsp = builder.build_model()

# Training
ddsp.train(dataset_path='path/to/dataset')

# Autoencoding
latents = ddsp.encode('path/to/audio.wav')
decoding = ddsp.decode(latents)
torchaudio.save('path/to/decoded_audio.wav', decoding, sample_rate=ddsp.fs)

# Export the model
ddsp.export('path/to/export/model.ts')
```

### Feature-based ddsp model (with features like loudness or pitch)
```python
builder = DDSPBuilder()

# Define the network
builder.add_encoder('ae', latent_size=128)
builder.add_synth_block('noiseband', n_filters=2048)
builder.add_feature('loudness')
builder.add_feature('spectral_centroid')
ddsp = builder.build_model()

# Training
ddsp.train(dataset_path='path/to/dataset')

# Export the model
ddsp.export('path/to/export/model.ts')
```
