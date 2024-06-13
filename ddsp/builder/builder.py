"""
Assembling the ddsp model from components

Example usage:

# VAE-based ddsp model (with latent space only)
builder = DDSPBuilder()
builder.add_encoder('vae', latent_size=8, beta=0.1)
builder.add_synth_block('noiseband', n_filters=512)
builder.add_synth_block('harmonic', n_harmonics=500)

ddsp = builder.build_model()
ddsp.train(dataset_path='path/to/dataset')
ddsp.export('path/to/export/model.ts')

# Feature-based ddsp model (with features like loudness or pitch)

# Combination of both
builder = DDSPBuilder()
builder.add_feature('loudness')
builder.add_feature('pitch')
builder.add_encoder('vae', latent_size=8, beta=0.1)
"""

from ddsp.ddsp import DDSP

class DDSPBuilder(object):
  def __init__(self):
    """
    """
    pass


  def add_encoder(self, encoder_type, **encoder_kwargs):
    """
    """

  def add_synth_block(self, synth_type, **kwargs):
    """
    """
    pass

  def add_feature(self, feature_name: str):
    """
    """
    pass

  def build_ddsp(self) -> DDSP:
    """
    """
    pass
