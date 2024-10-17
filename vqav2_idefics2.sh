#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="idefics2-base-8b-few-shots-8"
SPLIT="llava_vqav2_mscoco_test2015"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m eval.idefics2_vqa_loader \
#         --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
#         --image-folder ./playground/data/eval/vqav2/test2015 \
#         --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX &
# done

# CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m eval.idefics2_vqa_loader \
#     --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
#     --image-folder ./playground/data/eval/vqav2/test2015 \
#     --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#     --num-chunks $CHUNKS \
#     --chunk-idx 0

# wait

# output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

python convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

