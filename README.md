# FILTER: An Enhanced Fusion Method for Cross-lingual Language Understanding

This is the official repository of [FILTER](https://arxiv.org/abs/2009.05166).

## Requirements
We provide Docker image for easier reproduction. Please use `dockers/Dockerfile` or pull image directly.
```bash
docker pull studyfang/multilingual:xtreme
```

To run docker without sudo permission, please refer this documentation [Manage Docker as a non-root user](https://docs.docker.com/install/linux/linux-postinstall/).
Then, you could start docker, e.g.
```bash
docker run --gpus all -it -v /path/to/FILTER:/ssd -it studyfang/multilingual:xtreme bash
```

## Quick Start

**NOTE**: Please make sure you have set up the environment correctly. 

1. Download raw data and our preprocessed data. 

Please set your `DATA_ROOT` in init.sh, and then run the following command to download specified task and its pretrained FILTER models.
```bash
bash scripts/download_data.sh ${task}
```

To download all tasks and its pretrained models, please run `bash scripts/download_data.sh` which may take a while.


2. Evaluate our pretrained models which are save in `$DATA_ROOT/outputs/phase${idx}/${task}` :
```bash
bash eval.sh -t ${task} -n phase${idx}/${task}
```

where 
- `idx` could be `1` (without self-teaching) or `2`(+ self-teaching).
- `task` is the name of the task to evaluate from (`[xnli, pawsx, mlqa, tydiqa, xquad, udpos, panx]`)

## Model Training
For QA model training, we use translated training data from XTREME team. Please refere to their [repo](https://github.com/google-research/xtreme) or their [translation](https://console.cloud.google.com/storage/browser/xtreme_translations) directly.
Once your data is ready, simply run the following command to train a FILTER model for supported XTREME tasks:
```bash
bash train.sh -t ${task} -n ${task}
```
To use different number of local and fusion layers, you can run this command:
```bash
bash train.sh -t ${task} -n ${task}_k${k}_m${m}_ -x "--filter_k ${k} --filter_m ${m}"
```

where 
- `task` is the name of the task to train from (`[xnli, pawsx, mlqa, tydiqa, xquad, udpos, panx]`)
- `k` is the number of fusion layers
- `m` is the number of local layers

The output model will be save into `${DATA_ROOT}/outputs/${task}_k${k}_m${m}`.

**Note that we ran experiments on 8 V100 GPUs for FILTER models. You may need to increase `gradient_accumulation_steps` if you have less GPUs.**


## Citation
If you use this code useful, please star our repo or consider citing:
```
@article{fang2020filter,
  title={FILTER: An enhanced fusion method for cross-lingual language understanding},
  author={Fang, Yuwei and Wang, Shuohang and Gan, Zhe and Sun, Siqi and Liu, Jingjing},
  journal={arXiv preprint arXiv:2009.05166},
  year={2020}
}
```

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

MIT
