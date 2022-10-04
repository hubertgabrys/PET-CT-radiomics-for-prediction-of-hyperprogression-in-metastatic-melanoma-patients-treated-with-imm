#!/bin/sh

python model_run.py --modality ct
python model_run.py --modality pet
python model_run.py --modality petct

