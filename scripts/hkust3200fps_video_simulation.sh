#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export ABSL_VERBOSITY=1
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH


echo "Conda Enviroment CONDA_DEFAULT_ENV="$CONDA_DEFAULT_ENV

python hesim/simulator/hesim_for_hkust_3200fps_video_dataset.py
