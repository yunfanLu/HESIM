#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export ABSL_VERBOSITY=1


echo "Conda Enviroment CONDA_DEFAULT_ENV="$CONDA_DEFAULT_ENV

python  hesim/calibration/aps_calibrator.py