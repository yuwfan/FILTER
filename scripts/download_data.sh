#!/bin/bash

ALL_TASKS=${1:-"xnli pawsx mlqa xquad tydiqa panx udpos"}

source init.sh

BLOB='https://convaisharables.blob.core.windows.net/filter'
REPO=$PWD
DATA_DIR=$DATA_ROOT/data_raw
mkdir -p $DATA_DIR

# download XNLI dataset
function download_xnli {
    # download XNLI translations
    download_translations xnli

    OUTPATH=$DATA_DIR/xnli-tmp
    # download translate train data from official path
    if [ ! -d $OUTPATH/XNLI-MT-1.0 ]; then
        if [ ! -f $OUTPATH/XNLI-MT-1.0.zip ]; then
            wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip -P $OUTPATH -q --show-progress
        fi
        unzip -qq $OUTPATH/XNLI-MT-1.0.zip -d $OUTPATH
    fi
    if [ ! -d $OUTPATH/XNLI-1.0 ]; then
        if [ ! -f $OUTPATH/XNLI-1.0.zip ]; then
            wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip -P $OUTPATH -q --show-progress
        fi
        unzip -qq $OUTPATH/XNLI-1.0.zip -d $OUTPATH
    fi

    python $REPO/third_party/utils_preprocess.py \
           --data_dir $OUTPATH \
           --output_dir $DATA_DIR/xnli/ \
           --task xnli

    # rename for FILTER
	  for lang in ar bg de el en es fr hi ru sw th tr ur vi zh
	  do
        mv $OUTPATH/XNLI-MT-1.0/multinli/multinli.train.${lang}.tsv $DATA_DIR/xnli/train.en-${lang}.tsv
        mv $DATA_DIR/xnli/dev-${lang}.tsv $DATA_DIR/xnli/xtreme.${lang}.dev
        mv $DATA_DIR/xnli/test-${lang}.tsv $DATA_DIR/xnli/xtreme.${lang}.test
	  done

    rm -rf $OUTPATH
}

# download PAWS-X dataset
function download_pawsx {
    download_translations pawsx

    cd $DATA_DIR
    wget https://storage.googleapis.com/paws/pawsx/x-final.tar.gz -q --show-progress
    tar xzf x-final.tar.gz -C $DATA_DIR/

    # rename for FILTER
    mkdir -p $DATA_DIR/pawsx
    mv $DATA_DIR/x-final/en/train.tsv $DATA_DIR/pawsx/train.en-en.tsv
    for lang in de es fr ja ko zh
    do
        mv $DATA_DIR/x-final/$lang/translated_train.tsv $DATA_DIR/pawsx/train.en-${lang}.tsv
        mv $DATA_DIR/x-final/$lang/dev_2k.tsv $DATA_DIR/pawsx/xtreme.${lang}.dev
        mv $DATA_DIR/x-final/$lang/test_2k.tsv $DATA_DIR/pawsx/xtreme.${lang}.test
    done
    rm -rf x-final x-final.tar.gz
}

# download UD-POS dataset
function download_udpos {
    download_translations udpos

    base_dir=$DATA_DIR/udpos-tmp
    out_dir=$base_dir/conll/
    mkdir -p $out_dir

    mv $DATA_DIR/udpos/ $DATA_DIR/translations

    cd $base_dir
    curl -s --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/ud-treebanks-v2.5.tgz
    tar -xzf $base_dir/ud-treebanks-v2.5.tgz

    langs=(af ar bg de el en es et eu fa fi fr he hi hu id it ja kk ko mr nl pt ru ta te th tl tr ur vi yo zh)
    for x in $base_dir/ud-treebanks-v2.5/*/*.conllu; do
        file="$(basename $x)"
        IFS='_' read -r -a array <<< "$file"
        lang=${array[0]}
        if [[ " ${langs[@]} " =~ " ${lang} " ]]; then
            lang_dir=$out_dir/$lang/
            mkdir -p $lang_dir
            y=$lang_dir/${file/conllu/conll}
            if [ ! -f "$y" ]; then
                echo "python $REPO/third_party/ud-conversion-tools/conllu_to_conll.py $x $y --lang $lang --replace_subtokens_with_fused_forms --print_fused_forms"
                python $REPO/third_party/ud-conversion-tools/conllu_to_conll.py $x $y --lang $lang --replace_subtokens_with_fused_forms --print_fused_forms
            else
                echo "${y} exists"
            fi
        fi
    done

    python $REPO/third_party/utils_preprocess.py --data_dir $out_dir/ --output_dir $DATA_DIR/udpos/ --task  udpos
    rm -rf $out_dir ud-treebanks-v2.tgz $DATA_DIR/udpos-tmp
    echo "Successfully downloaded data at $DATA_DIR/udpos" >> $DATA_DIR/download.log

    # preprocess
    cd $REPO
    bash $REPO/scripts/preprocess_udpos.sh xlm-roberta-large $DATA_DIR

    mv $DATA_DIR/translations $DATA_DIR/udpos/udpos_processed_maxlen128/translations

}

function download_panx {
    
    echo "Download panx NER dataset"
    if [ -f $DATA_DIR/AmazonPhotos.zip ]; then
        download_translations panx
        mv $DATA_DIR/panx/ $DATA_DIR/translations

        base_dir=$DATA_DIR/panx_dataset/
        unzip -qq -j $DATA_DIR/AmazonPhotos.zip -d $base_dir
        cd $base_dir
        langs=(ar he vi id jv ms tl eu ml ta te af nl en de el bn hi mr ur fa fr it pt es bg ru ja ka ko th sw yo my zh kk tr et fi hu)
        for lg in ${langs[@]}; do
            tar xzf $base_dir/${lg}.tar.gz
            for f in dev test train; do mv $base_dir/$f $base_dir/${lg}-${f}; done
        done
        cd ..
        python $REPO/third_party/utils_preprocess.py \
            --data_dir $base_dir \
            --output_dir $DATA_DIR/panx \
            --task panx
        rm -rf $base_dir
        echo "Successfully downloaded data at $DATA_DIR/panx" >> $DATA_DIR/download.log

        cd $REPO
        bash $REPO/scripts/preprocess_panx.sh xlm-roberta-large $DATA_DIR

        mv $DATA_DIR/translations $DATA_DIR/panx/panx_processed_maxlen128/translations
    else
        echo "Please download the AmazonPhotos.zip file on Amazon Cloud Drive mannually and save it to $DATA_DIR/AmazonPhotos.zip"
        echo "https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN"
    fi


}

function download_squad {
    echo "download squad"
    base_dir=$DATA_DIR/squad/
    mkdir -p $base_dir && cd $base_dir
    wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json -q --show-progress
    wget https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json -q --show-progress
    echo "Successfully downloaded data at $DATA_DIR/squad"  >> $DATA_DIR/download.log
}

function download_xquad {
    # download translation
    download_translations xquad

    echo "download xquad"
    base_dir=$DATA_DIR/xquad/
    mkdir -p $base_dir && cd $base_dir
    for lang in ar de el en es hi ru th tr vi zh; do
      wget https://raw.githubusercontent.com/deepmind/xquad/master/xquad.${lang}.json -q --show-progress
    done
    python $REPO/third_party/utils_preprocess.py --data_dir $base_dir --output_dir $base_dir --task xquad
    echo "Successfully downloaded data at $DATA_DIR/xquad" >> $DATA_DIR/download.log
}

function download_mlqa {
    echo "download mlqa"
    base_dir=$DATA_DIR/mlqa/
    mkdir -p $base_dir && cd $base_dir
    zip_file=MLQA_V1.zip
    wget https://dl.fbaipublicfiles.com/MLQA/${zip_file} -q --show-progress
    unzip -qq ${zip_file}
    rm ${zip_file}
    python $REPO/third_party/utils_preprocess.py --data_dir $base_dir/MLQA_V1/test --output_dir $base_dir --task mlqa
    echo "Successfully downloaded data at $DATA_DIR/mlqa" >> $DATA_DIR/download.log
}

function download_tydiqa {
    download_translations tydiqa
    mv $DATA_DIR/tydiqa $DATA_DIR/translations

    echo "download tydiqa-goldp"
    base_dir=$DATA_DIR/tydiqa/
    mkdir -p $base_dir && cd $base_dir
    tydiqa_train_file=tydiqa-goldp-v1.1-train.json
    tydiqa_dev_file=tydiqa-goldp-v1.1-dev.tgz
    wget https://storage.googleapis.com/tydiqa/v1.1/${tydiqa_train_file} -q --show-progress
    wget https://storage.googleapis.com/tydiqa/v1.1/${tydiqa_dev_file} -q --show-progress
    tar -xf ${tydiqa_dev_file}
    rm ${tydiqa_dev_file}
    out_dir=$base_dir/tydiqa-goldp-v1.1-train
    python $REPO/third_party/utils_preprocess.py --data_dir $base_dir --output_dir $out_dir --task tydiqa
    mv $base_dir/$tydiqa_train_file $out_dir/
    echo "Successfully downloaded data at $DATA_DIR/tydiqa" >> $DATA_DIR/download.log

    mv $DATA_DIR/translations/* $DATA_DIR/tydiqa/tydiqa-goldp-v1.1-dev/
    rm -rf $DATA_DIR/translations
}

function download_translations() {
    task=$1
    wget ${BLOB}/data/translations/${task}.zip -O $DATA_DIR/${task}.zip
    unzip $DATA_DIR/${task}.zip -d $DATA_DIR
    rm $DATA_DIR/${task}.zip
}

function download_data() {
    task=$1

    case $task in
        xnli)
            download_xnli
            ;;
        pawsx)
            download_pawsx
            ;;
        mlqa)
            download_mlqa
            ;;
        tydiqa)
            download_tydiqa
            ;;
        xquad)
            download_xquad
            ;;
        udpos)
            download_udpos
            ;;
        panx)
            download_panx
            ;;
    esac
}

function download_model() {
    task=$1

    mkdir -p $DATA_ROOT/outputs/phase1/$task $DATA_ROOT/outputs/phase2/$task
    for file in config.json pytorch_model.bin sentencepiece.bpe.model special_tokens_map.json tokenizer_config.json; do
        wget ${BLOB}/data/outputs/phase1/${task}/$file -O $DATA_ROOT/outputs/phase1/$task/$file
        wget ${BLOB}/data/outputs/phase2/${task}/$file -O $DATA_ROOT/outputs/phase2/$task/$file
    done
}

for task in $ALL_TASKS
do
    download_data $task
    #download_model $task
done
