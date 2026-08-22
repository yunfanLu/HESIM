#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export ABSL_VERBOSITY=1
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

python -m unittest discover -s tests -p 'test_*.py'
