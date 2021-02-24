# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

BLOB='https://convaisharables.blob.core.windows.net/filter/data/outputs/phase2'

download_model() {
    mkdir -p $DOWNLOAD/outputs/phase2/$task
    for file in config.json pytorch_model.bin sentencepiece.bpe.model special_tokens_map.json tokenizer_config.json; do
        wget $BLOB/$task/$file -O $DOWNLOAD/outputs/phase2/$task/$file
    done
}
