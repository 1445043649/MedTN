#!/bin/bash

MODEL_NAME_OR_PATH='./chinese-bert-wwm-ext'
OUTPUT_DIR='./result/example'
DATA_DIR='./datasets/example'



CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_dictionary_path ${DATA_DIR}/train_dictionary.txt \
    --dictionary_path ${DATA_DIR}/train_dictionary.txt \
    --train_dir ${DATA_DIR}/processed_traindev \
    --data_dir ${DATA_DIR}/processed_test \
    --output_dir ${OUTPUT_DIR} \
    --use_cuda \
    --topk 20 \
    --epoch 30 \
    --train_batch_size 2 \
    --learning_rate 1e-5 \
    --sparse_learning_rate 1e-2 \
    --max_length 512
