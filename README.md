# VAE-controlled DDSP
This project is a pytorch implementation of DDSP based on the paper [DDSP: Differentiable Digital Signal Processing](https://arxiv.org/abs/2001.04643), customised for the generation of environmental sound.

The important differences include:
- latent space is regularised with a VAE loss and there is no explicit features extraction block. Therefore the control of generation is based on latent variables.
- the residual part is based on [NoiseBandNet](https://arxiv.org/abs/2307.08007) instead of a simple noise generator.
- sinusoidal modeling is used instead of the harmonic one for modeling of the tonal component.

## Installaation
Clone the repository and install the package locally with pip:
```bash
pip install -r requirements.txt
pip install -e .
```

## Training
The training is done in two steps:
1. Preprocess the dataset
```bash
python utils/dataset_converter.py --input_dir <path_to_dataset> --output_dir <path_to_output_dir>
```
2. Train the model
```bash
python cli/train.py\
        --latent_size 8\
        --model_name <model_name>\
        --dataset_path <path_to_preprocessed_dataset>
```

The training process is highly customisable. To see all the options run:
```bash
python cli/train.py --help
```
## Inference

### Max/MSP and PureData
The model is compatibile with nn~ externals for Max/MSP and PureData. In order to use trained model, you need to install the extensions following the instructions from [original nn~ repository](https://github.com/acids-ircam/nn_tilde).

### Model export
In order to export the model to be used with nn~ externals, run:
```bash
python cli/export.py --model_directory <path_to_model_training> --output_dir <path_to_output_dir>
```


