#!/bin/sh
python train_seven_branch.py --cuda_device 0 \
                --train_path   \
                --val_path   \
    --batch_size 128 \
		--epochs 70 \
		--class_num 7 \
		--train_name 1 \
		--l_r 0.000005 \
		--branch_size 256 \
		--recordname sfew
