#!/bin/bash

source init.sh

MODEL_NAME_OR_PATH='xlm-roberta-large'

usage()
{
cat << EOF
usage: $0 options
OPTIONS:
        -h      Show the help and exit
        -n      Experiment name to evaluate
        -m      Pretrained model name or path
        -t      task to evaluate
        -x      For convinent usage
EOF
}

while getopts "h:d:m:n:t:x:k:" opt
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
			          MODEL_NAME_OR_PATH=$OPTARG
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

OUTPUT_DIR=$DATA_ROOT/outputs/$EXP_NAME
if [[ ! -d $OUTPUT_DIR ]]; then
	  echo "$OUTPUT_DIR not exist, please specify it"
	  exit 1
fi

xnli() {
python ./examples/run_xcls.py \
       --task_name xnli \
       --model_type filter \
       --data_dir $DATA_DIR/xnli \
       --model_name_or_path $MODEL_NAME_OR_PATH \
       --language ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh \
       --train_language en \
       --do_eval \
       --eval_splits 'valid' \
       --max_seq_length 256 \
       --output_dir $OUTPUT_DIR \
       --per_gpu_eval_batch_size 64 \
       --filter_m 1 --filter_k 1 \
       ${OTHER_ARGS}
}

pawsx() {
python ./examples/run_xcls.py \
       --task_name pawsx \
       --data_dir $DATA_DIR/pawsx \
       --model_type filter \
       --model_name_or_path $MODEL_NAME_OR_PATH \
       --language de,en,es,fr,ja,ko,zh \
       --train_language en \
       --do_eval \
       --eval_splits valid \
       --max_seq_length 256 \
       --output_dir $OUTPUT_DIR \
       --per_gpu_eval_batch_size 64 \
       --filter_m 1 --filter_k 1 \
       ${OTHER_ARGS}
}

# mlqa and xquad share the same training set
mlqa() {
python ./examples/run_xqa.py \
       --task_name mlqa \
       --data_dir $DATA_DIR \
       --model_type filter \
       --model_name_or_path $MODEL_NAME_OR_PATH \
       --language en,es,de,ar,hi,vi,zh \
       --train_language en \
       --do_eval \
       --eval_splits dev \
       --do_lower_case \
       --per_gpu_eval_batch_size 64 \
       --max_seq_length 384 \
       --doc_stride 128  \
       --output_dir $OUTPUT_DIR \
       --threads 8 \
       --filter_m 1 --filter_k 20 \
       ${OTHER_ARGS}
}

xquad() {
    python ./examples/run_xqa.py \
           --task_name xquad \
           --model_type filter \
           --model_name_or_path $MODEL_NAME_OR_PATH \
           --do_eval \
           --eval_splits 'test' \
           --do_lower_case \
           --language ar,de,el,en,es,hi,ru,th,tr,vi,zh \
           --train_language en \
           --data_dir $DATA_DIR \
           --per_gpu_eval_batch_size 64 \
           --max_seq_length 384 \
           --doc_stride 128  \
           --output_dir $OUTPUT_DIR \
           --threads 8 \
           --filter_m 1 --filter_k 20 \
           ${OTHER_ARGS}
}

tydiqa() {
python ./examples/run_xqa.py \
       --task_name tydiqa \
       --model_type filter \
       --model_name_or_path $MODEL_NAME_OR_PATH \
       --do_eval \
       --do_lower_case \
       --language ar,bn,en,fi,id,ko,ru,sw,te \
       --eval_splits dev \
       --train_language en \
       --data_dir $DATA_DIR \
       --per_gpu_eval_batch_size 64 \
       --max_seq_length 384 \
       --doc_stride 128  \
       --output_dir $OUTPUT_DIR \
       --threads 8 \
       ${OTHER_ARGS}
}

udpos() {
    python ./examples/run_xtag.py \
           --task_name udpos \
           --model_type filter \
           --data_dir $DATA_DIR/udpos/udpos_processed_maxlen128 \
           --labels $DATA_DIR/udpos/udpos_processed_maxlen128/labels.txt \
           --model_name_or_path $MODEL_NAME_OR_PATH \
           --output_dir $OUTPUT_DIR \
           --train_language en \
           --language 'af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,mr,nl,pt,ru,ta,te,th,tl,tr,ur,vi,yo,zh' \
           --eval_splits dev \
           --max_seq_length 128 \
           --per_gpu_eval_batch_size 64 \
           --do_eval \
           --filter_m 1 --filter_k 20 \
           ${OTHER_ARGS}
}

panx() {
python ./examples/run_xtag.py \
       --task_name panx \
       --model_type filter \
       --data_dir $DATA_DIR/panx/panx_processed_maxlen128 \
       --labels $DATA_DIR/panx/panx_processed_maxlen128/labels.txt \
       --model_name_or_path $MODEL_NAME_OR_PATH \
       --output_dir $OUTPUT_DIR \
       --language ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu \
       --eval_splits dev \
       --train_language en \
       --max_seq_length 128 \
       --per_gpu_eval_batch_size 64 \
       --do_eval \
       ${OTHER_ARGS}
}


for task in xnli pawsx mlqa xquad tydiqa panx udpos; do
        if [[ ${TASK:-"xnli"} == $task ]]; then
                $task
        fi
done
