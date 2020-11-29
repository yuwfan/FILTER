#!/bin/bash

source init.sh

MODEL_NAME_OR_PATH='xlm-roberta-large'

usage()
{
cat << EOF
usage: $0 options
OPTIONS:
        -h      Show the help and exit
        -n      Experiment name for saving to output directory
        -m      Pretrained model name or path
        -g      gpus to use, default is to use all GPUs
        -t      task to train
        -x      For convinent usage
EOF
}

while getopts "h:m:n:g:t:x:" opt
do
        case $opt in
            h)
                usage
                exit 1
                ;;
		        n)
			          EXP_NAME=$OPTARG
			          ;;
		        m)
			          MODEL_NAME_OR_PATH=${OPTARG}
			          ;;
		        g)
			          N_GPU=$OPTARG
			          ;;
		        t)
			          TASK=$OPTARG
			          ;;
            x)
                OTHER_ARGS=$OPTARG
                ;;
        esac
done

DATA_DIR=$DATA_ROOT/data_raw
if [[ ! -d $DATA_DIR ]]; then
	  echo "$DATA_DIR not exist"
	  exit 1
fi


OUTPUT_DIR=$DATA_ROOT/outputs/${EXP_NAME:-debug}
mkdir -p $OUTPUT_DIR

xnli() {
    python -m torch.distributed.launch --nproc_per_node=$N_GPU --master_port=$RANDOM ./examples/run_xcls.py \
       --task_name xnli \
       --data_dir $DATA_DIR/xnli \
       --model_type filter \
       --model_name_or_path $MODEL_NAME_OR_PATH \
       --language ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh \
       --train_language en \
       --do_train \
       --eval_splits valid \
       --fp16 \
       --per_gpu_train_batch_size 8 \
       --learning_rate 3e-6 \
       --num_train_epochs 5 \
       --max_seq_length 256 \
       --output_dir $OUTPUT_DIR \
       --log_dir $OUTPUT_DIR \
       --overwrite_output_dir \
       --logging_steps 500 \
       --logging_each_epoch \
       --per_gpu_eval_batch_size 64 \
       --eval_all_checkpoints \
       --filter_m 1 --filter_k 1 \
       ${OTHER_ARGS}
}

pawsx() {
    python -m torch.distributed.launch --nproc_per_node=$N_GPU --master_port=$RANDOM ./examples/run_xcls.py \
       --task_name pawsx \
       --data_dir $DATA_DIR/pawsx \
       --model_type filter \
       --language de,en,es,fr,ja,ko,zh \
       --model_name_or_path $MODEL_NAME_OR_PATH \
       --train_language en \
       --do_train \
       --eval_splits valid \
       --per_gpu_train_batch_size 4 \
       --learning_rate 1e-5 \
       --num_train_epochs 4 \
       --max_seq_length 256 \
       --output_dir $OUTPUT_DIR \
       --log_dir $OUTPUT_DIR \
       --overwrite_output_dir \
       --logging_steps 500 \
       --per_gpu_eval_batch_size 64 \
       --logging_each_epoch \
       --filter_m 1 --filter_k 1 \
       ${OTHER_ARGS}
}

mlqa() {
    # mlqa and xquad share the same training set
    python -m torch.distributed.launch --nproc_per_node=$N_GPU --master_port=$RANDOM ./examples/run_xqa.py \
           --task_name mlqa \
           --data_dir $DATA_DIR \
           --model_type filter \
           --model_name_or_path $MODEL_NAME_OR_PATH \
           --language ar,de,en,es,hi,vi,zh \
           --train_language en \
           --do_train \
           --eval_splits 'dev' \
           --do_lower_case \
           --per_gpu_train_batch_size 4 \
           --gradient_accumulation_steps 2 \
           --learning_rate 5e-6 \
           --per_gpu_eval_batch_size 64 \
           --num_train_epochs 2.0 \
           --max_seq_length 384 \
           --doc_stride 128  \
           --output_dir $OUTPUT_DIR \
           --log_dir $OUTPUT_DIR \
           --logging_each_epoch \
           --evaluate_during_training \
           --threads 8 \
           --filter_m 1 --filter_k 20 \
           ${OTHER_ARGS}
}


xquad() {
    python -m torch.distributed.launch --nproc_per_node=$N_GPU --master_port=$RANDOM ./examples/run_xqa.py \
       --task_name xquad \
       --data_dir $DATA_DIR/ \
       --model_type filter \
       --model_name_or_path $MODEL_NAME_OR_PATH \
       --language ar,de,el,en,es,hi,ru,th,tr,vi,zh \
       --train_language en \
       --do_train \
       --eval_splits 'dev' \
       --do_lower_case \
       --per_gpu_train_batch_size 4 \
       --learning_rate 5e-6 \
       --per_gpu_eval_batch_size 64 \
       --num_train_epochs 2.0 \
       --max_seq_length 384 \
       --doc_stride 128  \
       --output_dir $OUTPUT_DIR \
       --log_dir $OUTPUT_DIR \
       --logging_each_epoch \
       --eval_all_checkpoints \
       --threads 8 \
       --filter_m 1 --filter_k 20 \
       ${OTHER_ARGS}
}

tydiqa() {
    python -m torch.distributed.launch --nproc_per_node=$N_GPU --master_port=$RANDOM ./examples/run_xqa.py \
       --task_name tydiqa \
       --data_dir $DATA_DIR \
       --model_type filter \
       --model_name_or_path $MODEL_NAME_OR_PATH \
       --language ar,bn,en,fi,id,ko,ru,sw,te \
       --train_language en \
       --do_train \
       --do_lower_case \
       --eval_splits dev \
       --per_gpu_train_batch_size 4 \
       --learning_rate 1e-5 \
       --per_gpu_eval_batch_size 64 \
       --num_train_epochs 4.0 \
       --logging_each_epoch \
       --max_seq_length 384 \
       --doc_stride 128  \
       --output_dir $OUTPUT_DIR \
       --log_dir $OUTPUT_DIR \
       --overwrite_output_dir \
       --eval_all_checkpoints \
       --threads 8 \
       --filter_m 1 --filter_k 20 \
       ${OTHER_ARGS}
}

udpos() {
    python -m torch.distributed.launch --nproc_per_node=$N_GPU --master_port=$RANDOM ./examples/run_xtreme_tag.py \
           --task_name udpos \
           --data_dir $DATA_ROOT/udpos/udpos_processed_maxlen128 \
           --model_type filter \
           --model_name_or_path $MODEL_NAME_OR_PATH \
           --labels $DATA_ROOT/udpos/udpos_processed_maxlen128/labels.txt \
           --language af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,mr,nl,pt,ru,ta,te,th,tl,tr,ur,vi,yo,zh \
           --train_language en \
           --do_train \
           --eval_splits dev \
           --max_seq_length 128 \
           --num_train_epochs 20 \
           --per_gpu_train_batch_size 8 \
           --per_gpu_eval_batch_size 64 \
           --learning_rate 5e-6 \
           --save_steps 1000 \
           --output_dir $OUTPUT_DIR \
           --log_dir $OUTPUT_DIR \
           --eval_all_checkpoints \
           --filter_m 1 --filter_k 1 \
           ${OTHER_ARGS}
}

panx() {
    python -m torch.distributed.launch --nproc_per_node=${N_GPU:-8} --master_port=$RANDOM ./examples/run_tag.py \
           --task_name panx \
           --data_dir $DATA_ROOT/panx/panx_processed_maxlen128 \
           --labels $DATA_ROOT/panx/panx_processed_maxlen128/labels.txt \
           --model_type filter \
           --model_name_or_path $MODEL_NAME_OR_PATH \
           --language ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu \
           --train_language en \
           --do_train \
           --eval_splits dev \
           --max_seq_length 128 \
           --num_train_epochs 20 \
           --per_gpu_train_batch_size 8 \
           --per_gpu_eval_batch_size 64 \
           --learning_rate 5e-6 \
           --save_steps 1000 \
           --eval_all_checkpoints \
           --log_dir $OUTPUT_DIR \
           --output_dir $OUTPUT_DIR \
           --filter_m 1 --filter_k 1 \
           ${OTHER_ARGS}
}

for task in xnli pawsx mlqa xquad tydiqa udpos panx
do
        if [[ ${TASK:-"xnli"} == $task ]]; then
                $task
        fi
done
