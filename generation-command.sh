#!/bin/bash

python random_generate.py --model_checkpoint training/training/footsteps/version_0/checkpoints/epoch=9999-step=40000.ckpt --audio_duration 20 --save_path generation.wav
