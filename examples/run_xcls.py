# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning multi-lingual models on XNLI (Bert, DistilBERT, XLM).
    Adapted from `examples/run_glue.py`"""


import argparse
import glob
import logging
import os
import random
import json
import copy
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    FilterForSequenceClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import xglue_convert_examples_to_features as convert_examples_to_features
from transformers import xglue_compute_metrics as compute_metrics
from transformers import xglue_output_modes as output_modes
from transformers import xglue_processors as processors


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, DistilBertConfig, XLMConfig, XLMRobertaConfig)), ()
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "xlmr": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "filter": (XLMRobertaConfig, FilterForSequenceClassification, XLMRobertaTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class AlignDataset(Dataset):
    def __init__(self, datasets, english_index, is_training=True, use_all_samples='False'):
        super(AlignDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        assert len(set([len(dataset) for dataset in datasets])) == 1, "all dataset should have same length"
        self.datasets = datasets
        self.english_index = english_index
        self.is_training = is_training
        self.dataset_len = len(datasets[0])

        self.max_samples = self.dataset_len if not use_all_samples else len(self.datasets) * self.dataset_len
        self.sample_idx = [] if use_all_samples else self.get_sample_idx()


    def __len__(self):
        return self.max_samples if self.is_training else self.dataset_len

    def __getitem__(self, idx):
        # Order as datasets
        if self.is_training and self.sample_idx:
            idx = self.sample_idx[idx]

        lang_idx, offset = divmod(idx, self.dataset_len)
        return [self.datasets[lang_idx][offset], self.datasets[self.english_index][offset]]

    def get_sample_idx(self):
        dataset_idx = [random.choice(range(len(self.datasets))) for _ in range(self.dataset_len)]
        return [dataset_idx[i]*self.dataset_len + i for i in range(self.dataset_len)]

    def shuffle(self):
        self.sample_idx = self.get_sample_idx()

def get_max_steps(output_dir):
    checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
    max_step = 0
    for checkpoint in checkpoints:
        try:
            if len(checkpoint.split("-")) > 1:
                max_step = max(max_step, int(checkpoint.split("-")[-1]))
        except:
            continue
    return max_step


def train(args, train_dataset, label_list, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        if args.log_dir:
            tb_writer = SummaryWriter(args.log_dir)
        else:
            tb_writer = SummaryWriter()
        log_writer = open(os.path.join(args.output_dir, "evaluate_logs.txt"), 'a')

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
                )

    # model recover
    recover_step = get_max_steps(args.output_dir)
    model_recover_path = None
    if recover_step:
        model_recover_path = os.path.join(args.output_dir, "checkpoint-{}".format(recover_step))
        logger.info(" ** Recover model checkpoint in %s ** ", model_recover_path)
        model.load_state_dict(torch.load(os.path.join(model_recover_path, WEIGHTS_NAME), map_location='cpu'))
        model.to(args.device)

        # check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(model_recover_path, "optimizer.pt")) and os.path.isfile(
                    os.path.join(model_recover_path, "scheduler.pt")
                        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(model_recover_path, "optimizer.pt"), map_location='cpu'))
            scheduler.load_state_dict(torch.load(os.path.join(model_recover_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if model_recover_path and os.path.exists(model_recover_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = recover_step
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss, best_avg = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility

    def logging():
        results = evaluate(args, model, tokenizer, label_list, single_gpu=True, splits=args.eval_splits.split(','))
        for task, result in results.items():
            for key, value in result.items():
                tb_writer.add_scalar("eval_{}_{}".format(task, key), value, global_step)
        log_writer.write("{0}\t{1}\n".format(global_step, json.dumps(results)))
        log_writer.flush()
        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
        return results

    def save_checkpoint(cur_step):
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(cur_step))
        logger.info("Saving model checkpoint to %s", output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))

        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)


    for epc in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        if not args.use_all_samples_per_epoch:
            if args.local_rank != -1:
                train_sampler.set_epoch(epc)
            if epc > 0:
                train_dataset.shuffle()

        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            if args.filter_k == 0:
                # no cross-attention layer
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            else:
                input_ids = [d[0].to(args.device)  for d in batch]
                attention_mask = [d[1].to(args.device) for d in batch]
                labels = [d[3].to(args.device) for d in batch][0]
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

                if args.alpha > 0:
                    soft_labels = [d[4].to(args.device) for d in batch][0]
                    inputs["soft_labels"] = soft_labels

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert"] else None
                )  # XLM and DistilBERT don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(global_step)

                    if args.evaluate_during_training:
                        cur_result = logging()
                        logger.info(cur_result)


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.local_rank in [-1, 0] and args.logging_each_epoch:
            logging_loss = tr_loss
            save_checkpoint(global_step)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
        log_writer.close()

    return global_step, tr_loss / (global_step + 1)

def evaluate(args, model, tokenizer, label_list, prefix="", single_gpu=False, splits=['valid'], verbose=True):
    if single_gpu:
        args = copy.deepcopy(args)
        args.local_rank = -1
        args.n_gpu = 1
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    eval_datasets = []
    eval_langs = args.language.split(',')
    for split in splits:
        for lang in eval_langs:
            eval_datasets.append((split, lang))
    results = {}
    preds_dict = {}

    # leave interface for multi-task evaluation
    eval_task = eval_task_names[0]
    eval_output_dir = eval_outputs_dirs[0]

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    for split, lang in eval_datasets:
        task_name = "{0}-{1}".format(split, lang)
        if args.filter_k > 0:
            def get_pair_langs(args, split, lang):
                if split == 'train':
                    lang_pairs = ["en-{}".format(lang) if args.task_name == 'pawsx' and lang != 'en' else lang, 'en']
                else:
                    lang_pairs = [lang, "en" if lang == "en" else "msft.{}-en".format(lang)]
                return lang_pairs
            eval_langs = get_pair_langs(args, split, lang)

            dataset_list, guids = [], None
            for lg in eval_langs:
                lg_dataset, lg_guids = load_and_cache_examples(args, eval_task, tokenizer, lg, split=split)
                dataset_list.append(lg_dataset)
                if guids is None:
                    guids = lg_guids
            eval_dataset = AlignDataset(dataset_list, 1, is_training=False)
        else:
            eval_dataset, guids = load_and_cache_examples(args, eval_task, tokenizer, lang, split=split)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        preds = None
        out_label_ids = None
        guids = np.array(guids)
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            if args.filter_k == 0:
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                cur_labels = batch[3].detach().cpu().numpy()
            else:
                input_ids = [d[0].to(args.device)  for d in batch]
                attention_mask = [d[1].to(args.device) for d in batch]
                labels = [d[3].to(args.device) for d in batch][0]
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
                cur_labels = labels.detach().cpu().numpy()

            with torch.no_grad():
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids
                outputs = model(**inputs)
                logits = outputs[0]
                if args.filter_k > 0:
                    logits = logits[:, 0, :].squeeze(1)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = cur_labels
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, cur_labels, axis=0)

        if args.eval_train and split == 'train':
            dump_file = os.path.join(args.pkl_dir, '{}.{}.logits.pkl{}'.format(args.task_name, lang, args.pkl_index))
            pickle.dump(preds, open(dump_file, 'wb'), pickle.HIGHEST_PROTOCOL)
            continue

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        else:
            raise ValueError("No other `output_mode` for XGLUE.")
        if split not in preds_dict:
            preds_dict[split] = {}
        preds_dict[split][lang] = preds

        result = compute_metrics(eval_task, preds, out_label_ids, guids)
        results[task_name] = result

        results["{}_avg".format(split)] = average_dic([value for key, value in results.items() if key.startswith(split)])

    for split in splits:
        if split == 'train':
            continue

        for lang in preds_dict[split].keys():
            output_eval_file = os.path.join(args.output_dir, prefix, "{}-{}.tsv".format(split, lang))
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                for item in preds_dict[split][lang]:
                    writer.write(str(label_list[item]) + "\n")

    return results


def average_dic(dic_list):
    if len(dic_list) == 0:
        return {}
    dic_sum = {}
    for dic in dic_list:
        if len(dic_sum) == 0:
            for key, value in dic.items():
                dic_sum[key] = value
        else:
            assert set(dic_sum.keys()) == set(dic.keys()), "sum_keys:{0}, dic_keys:{1}".format(set(dic_sum.keys()), set(dic.keys()))
            for key, value in dic.items():
                dic_sum[key] += value
    for key in dic_sum:
        dic_sum[key] /= len(dic_list)
    return dic_sum


def load_and_cache_examples(args, task, tokenizer, language, split="train"):
    assert split in ["train", "valid", "dev", "test"]
    if args.local_rank not in [-1, 0] and split == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task](language=language, train_language=language, benchmark=args.benchmark)
    label_list = processor.get_labels()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    data_cache_name = list(filter(None, args.model_name_or_path.split("/"))).pop()
    if args.data_cache_name is not None:
        data_cache_name = args.data_cache_name
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}".format(
            split,
            data_cache_name,
            str(args.max_seq_length),
            str(task),
            str(language),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if split == "test":
            examples = processor.get_test_examples(args.data_dir)
        elif split in ["valid", 'dev']:
            examples = processor.get_valid_examples(args.data_dir)
        else:  # train
            examples = processor.get_train_examples(args.data_dir)

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and split == 'train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_guids = [f.guid for f in features]
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
        all_labels = torch.tensor([0 for f in features], dtype=torch.long)


    if split == 'train' and args.alpha > 0:
        # create new features and dataset
        language = language.replace("en-", "")
        if args.use_eng_logits:
            logits_files = [os.path.join(args.pkl_dir, "{}.{}.logits.pkl{}".format(args.task_name, 'en', idx))
                            for idx in args.pkl_index.split(',')]
        else:
            logits_files = [os.path.join(args.pkl_dir, "{}.{}.logits.pkl{}".format(args.task_name, language, idx))
                            for idx in args.pkl_index.split(',')]
        assert len(logits_files) > 0
        logger.info("Read logits files: {}".format(logits_files))

        soft_labels = np.zeros((all_input_ids.size(0), len(label_list)))
        for f in logits_files:
            pkl_data = pickle.load(open(f, 'rb'))
            assert len(pkl_data) == all_input_ids.size(0)
            soft_labels += pkl_data
        soft_labels /= len(logits_files)
        all_soft_labels = torch.from_numpy(soft_labels)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_soft_labels)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return dataset, all_guids

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--pkl_dir",
        default=None,
        type=str,
        help="The pkl data dir for training set logits.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--data_cache_name",
        default=None,
        type=str,
        help="The name of cached data",
    )
    parser.add_argument(
        "--language",
        default=None,
        type=str,
        required=True,
        help="Evaluation language. Also train language if `train_language` is set to None.",
    )
    parser.add_argument(
        "--benchmark", default='xtreme', type=str, help="benchmark, xglue/xtreme")
    parser.add_argument(
        "--train_language", default=None, type=str, help="Train language if is different of the evaluation language."
    )
    parser.add_argument(
        "--sample_ratio", default=0.0, type=float, help="The training sample ratio of each language"
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # Other parameters
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="The output log dir."
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--gpu_id", default=None, type=str, help="GPU id"
    )
    parser.add_argument("--filter_k", type=int, default=0)
    parser.add_argument("--filter_m", type=int, default=0)
    parser.add_argument("--first_loss_only", action='store_true')
    parser.add_argument("--use_eng_logits", action='store_true', help='use english soft logits for other language')

    parser.add_argument("--alpha", type=float, default=0, help='alpha for kd loss')
    parser.add_argument("--temperature", type=float, default=0.1, help="temprature to soft logits")

    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--hidden_dropout_prob",
        default=0.1,
        type=float,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the test set.")
    parser.add_argument("--eval_checkpoints", type=str, default=None, help="evaluation checkpoints")
    parser.add_argument("--eval_splits", default='valid', type=str, help="eval splits")
    parser.add_argument("--eval_train", action='store_true', help="eval splits")
    parser.add_argument("--pkl_index", default="0", type=str, help="pickle index for dumping training logits")

    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--use_all_samples_per_epoch", type=boolean_string, default='true', help="Use all samples for per epoch training"
    )
    parser.add_argument(
        "--max_train_samples_per_epoch", default=None, type=int, help="Total number of training epochs to perform."
    )

    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=-1, help="Log every X updates steps.")
    parser.add_argument("--logging_each_epoch", action="store_true", help="Whether to log after each epoch.")
    parser.add_argument("--logging_steps_in_sample", type=int, default=-1, help="log every X samples.")
    parser.add_argument("--save_steps", type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=52, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(

            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    if args.pkl_dir is None:
        args.pkl_dir = args.data_dir

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    logger.info("Training/evaluation parameters %s", args)

    # preprocess args
    assert not (args.logging_steps != -1 and args.logging_steps_in_sample != -1), "these two parameters can't both be setted"
    if args.logging_steps == -1 and args.logging_steps_in_sample != -1:
        total_batch_size = args.n_gpu * args.per_gpu_train_batch_size * args.gradient_accumulation_steps
        args.logging_steps = args.logging_steps_in_sample // total_batch_size

    # Set seed
    set_seed(args)

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name](language=args.language,
                                           train_language=args.train_language,
                                           benchmark=args.benchmark)
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.filter_k > 0: # there is cross attention layer
        config.first_loss_only = args.first_loss_only
        config.alpha = args.alpha
        config.temperature = args.temperature
        config.filter_m = args.filter_m
        config.hidden_dropout_prob = args.hidden_dropout_prob

        config.output_hidden_states = True
        config.filter_k = min(args.filter_k, config.num_hidden_layers - args.filter_m)
        config.num_hidden_layers = args.filter_m

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        filter_m=args.filter_m,
        filter_k=args.filter_k,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    # Training
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        if args.filter_k > 0: # FILTER
            train_langs = ["en-{}".format(lang) for lang in args.language.split(',')]
        else:
            train_langs = args.train_language.split(',')

        dataset_list = []
        for lang in train_langs:
            lg_train_dataset, guids = load_and_cache_examples(args, args.task_name, tokenizer, lang, split="train")
            dataset_list.append(lg_train_dataset)
        if args.filter_k > 0:
            train_dataset = AlignDataset(dataset_list,
                                         train_langs.index('en-en'),
                                         is_training=True,
                                         use_all_samples=args.use_all_samples_per_epoch)

        else:
            train_dataset = ConcatDataset(dataset_list)

        global_step, tr_loss = train(args, train_dataset, label_list, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        results = {}
        if args.eval_checkpoints:
            checkpoints = [os.path.join(args.output_dir, ckpt) for ckpt in args.eval_checkpoints.split(',')]
        else:
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        best_avg, best_checkpoint = 0, None
        task_metric = "acc" if args.task_name != "rel" else "ndcg"
        for checkpoint in checkpoints:
            prefix = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""

            tokenizer = tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, label_list, prefix=prefix, splits=args.eval_splits.split(','))
            results[os.path.basename(checkpoint)] = result

            logger.info("{}\t{}".format(checkpoint, result))

            if best_avg < result["valid_avg"][task_metric]:
                best_avg = result["valid_avg"][task_metric]
                best_checkpoint = checkpoint

        with open(os.path.join(args.output_dir, "eval_logs.txt"), 'w') as log_writer:
            for key, val in results.items():
                log_writer.write("{}\t{}\n".format(key, json.dumps(val)))

    if args.eval_train and args.local_rank in [-1, 0]:
        if args.eval_checkpoints:
            # use the first one
            checkpoint = [os.path.join(args.output_dir, ckpt) for ckpt in args.eval_checkpoints.split(',')][0]
        else:
            checkpoint = os.path.join(args.output_dir, 'checkpoint-best')

        assert os.path.exists(checkpoint)
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        evaluate(args, model, tokenizer, label_list, prefix="", splits=['train'])

    logger.info("Task {0} finished!".format(args.task_name))

if __name__ == "__main__":
    main()
