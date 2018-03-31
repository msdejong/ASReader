# AS Reader

## Introduction

This is a PyTorch implementation of the AS Reader as presented in "Text Understanding with the Attention Sum Reader Network" available at https://arxiv.org/abs/1603.01547.

We evaluate AS Reader on the CNN dataset (https://arxiv.org/abs/1506.03340)

## Quick Start
To run the model:
Download the data from https://cs.nyu.edu/~kcho/DMQA/. 

To train the model run the model with following arguments:-
```
python main.py 
             --train_path <path_to_training_folder>
             --valid_path <path_to_validation_folder>
             --test_path <path_to_test_folder>
             --eval_interval 500 
             --use_cuda True 
             --num_epochs 2 
             --learning_rate 0.001 
             --seed 2  
```
