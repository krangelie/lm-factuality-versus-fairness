#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

nohup python run_experiment.py benchmark=bold model=gpt2 > /export/home/kraft/data/bold/bold_generated_sentences/gpt2/log.out 2>&1 &
