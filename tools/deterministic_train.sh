#!/usr/bin/env bash

CONFIG=$1
python tools/train.py --config $CONFIG --seed 0 --deterministic
