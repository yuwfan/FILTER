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
""" Fine-tuning the library models for named entity recognition """

import argparse
import glob
import logging
import os
import random
import json
import numpy as np
import torch
import pickle
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaForTokenClassification,
    FilterForTokenClassification,
    get_linear_schedule_with_warmup,
)
from utils_tag import get_labels, convert_examples_to_features, read_examples_from_file

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (XLMRobertaConfig,)
    ),
    (),
)

MODEL_CLASSES = {
    "xlmr": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
    "filter": (XLMRobertaConfig, FilterForTokenClassification, XLMRobertaTokenizer),
}

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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

def train(args, train_dataset, model, tokenizer, labels_list, pad_token_label_id):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        if args.log_dir:
            tb_writer = SummaryWriter(args.log_dir)
        else:
            tb_writer = SummaryWriter()
        log_writer = open(os.path.join(args.output_dir, "train_eval_logs.txt"), 'w')

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
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
    if args.recover_model and recover_step:
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

    tr_loss, logging_loss, best_metric = 0.0, 0.0, 0.0
    best_preds = {}
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility

    def logging():
        results, preds_results = evaluate(args,
                                          model,
                                          tokenizer,
                                          labels_list,
                                          pad_token_label_id,
                                          eval_splits=args.eval_splits.split(','))
        for task, result in results.items():
            for key, value in result.items():
                tb_writer.add_scalar("eval_{}_{}".format(task, key), value, global_step)
        log_writer.write("{0}\t{1}\n".format(global_step, json.dumps(results)))
        log_writer.flush()
        measure_metric = "f1"
        if args.benchmark == 'xglue' and args.task_name == 'pos':
            measure_metric = "acc"
        return results["dev_avg"][measure_metric], preds_results

    # Save model checkpoint
    def save_checkpoint(cur_step):
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(cur_step))
        logger.info("Saving model checkpoint to %s", output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

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
                labels = [d[3].to(args.device) for d in batch]
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
                if args.alpha > 0:
                    soft_labels = [d[4].to(args.device) for d in batch]
                    inputs["soft_labels"] = soft_labels

            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

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

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
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
                        avg_metric, preds_results = logging()
                        logger.info(avg_metric)


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.local_rank in [-1, 0] and args.logging_each_epoch:
            save_checkpoint(global_step)

            if args.evaluate_during_training:
                avg_metric, preds_results = logging()
                logger.info(avg_metric)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
        log_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels_list, pad_token_label_id, prefix="", eval_splits=['dev']):
    eval_datasets = []
    eval_langs = args.language.split(',')
    for split in eval_splits:
        for lang in eval_langs:
            eval_datasets.append((split, lang))
    results = {}
    preds_results = {}
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    for split, lang in eval_datasets:
        task_name = "{0}-{1}".format(split, lang)
        if args.filter_k > 0:
            def get_pair_langs(args, split, lang):
                if split == 'train':
                    lang_pairs = ["en-{}".format(lang) if lang != 'en' else lang, 'en']
                else:
                    lang_pairs = [lang, "en" if lang == "en" else "{}-en".format(lang)]
                return lang_pairs
            eval_langs = get_pair_langs(args, split, lang)

            dataset_list = []
            for lg in eval_langs:
                # TODO: if no Eng or dev data not exist?
                lg_dataset = load_and_cache_examples(args, tokenizer, labels_list, pad_token_label_id, mode=split, lang=lg)
                if len(dataset_list) == 0 and lg_dataset is None: # data not exist
                    break
                #if len(dataset_list) == 1 and (lg_dataset is None or lang in ['zh', 'ja', 'th']): # no translation or no space for target language
                if len(dataset_list) == 1 and (lg_dataset is None):
                    dataset_list.append(dataset_list[0])
                else:
                    dataset_list.append(lg_dataset)

            if len(dataset_list) == 0:
                assert split != 'test', "You must predidct on {} for test".format(lang)
                continue
            eval_dataset = AlignDataset(dataset_list, 1, is_training=False)
        else:
            eval_dataset = load_and_cache_examples(args, tokenizer, labels_list, pad_token_label_id, mode=split, lang=lang)

        if not eval_dataset:
            assert split != 'test', "You must predidct on {} for test".format(lang)
            logger.info("Skip {} {}".format(split, lang))
            continue

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank in [-1, 0] else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation %s *****", prefix)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        preds = None
        out_label_ids = None
        model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            if args.filter_k == 0:
                batch = tuple(t.to(args.device) for t in batch)
                labels = batch[3]
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            else:
                input_ids = [d[0].to(args.device)  for d in batch]
                attention_mask = [d[1].to(args.device) for d in batch]
                all_labels = [d[3].to(args.device) for d in batch]
                labels = all_labels[0]
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

            with torch.no_grad():
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids
                outputs = model(**inputs)
                logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        if args.eval_train and split == 'train':
            dump_file = os.path.join(args.pkl_dir, '{}.{}.logits.pkl{}'.format(args.task_name, lang, args.pkl_index))
            pickle.dump(preds, open(dump_file, 'wb'), pickle.HIGHEST_PROTOCOL)
            continue

        preds = np.argmax(preds, axis=2)
        label_map = {i: label for i, label in enumerate(labels_list)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        result = {
            "precision": precision_score(out_label_list, preds_list),
            "acc": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

        results[task_name] = result
        preds_results[task_name] = preds_list

    for split in eval_splits:
        if split != 'train':
            results["{}_avg".format(split)] = average_dic([value for key, value in results.items() if key.startswith(split)])

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_results


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train", lang="en", use_barrier=True):
    if use_barrier and args.local_rank not in [-1, 0] and mode == 'train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}".format(
            args.task_name, mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length),
            lang
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:

        if '-' in lang: # hack for now: translation
            trans_file= os.path.join(args.data_dir, 'translations', "{}-{}.tsv".format(mode, lang))
            lang_fmt = lang.replace("-en", "") if mode != 'train' else 'en'
            if not os.path.exists(trans_file):
                logger.info("{} not exist".format(trans_file))
                return None
        else:
            lang_fmt = lang
            trans_file=None

        #file_path = os.path.join(args.data_dir, "{}-{}.tsv".format(mode, lang_fmt))
        file_path = os.path.join(args.data_dir,
                                 lang_fmt,
                                 "{}.{}".format(mode, os.path.basename(args.model_name_or_path)))

        if not os.path.exists(file_path):
            logger.info("{} not exist".format(file_path))
            return None

        examples = read_examples_from_file(file_path, lang, trans_file, split_no_space_lang=args.task_name in ['panx', 'ner'])
        logger.info("Creating features from dataset file at {} in language {}".format(file_path, lang))
        if not examples: # file may not exist:
            return None

        features = convert_examples_to_features(
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            lang,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta", "xlmroberta", "interxlmroberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if mode == 'train' and args.alpha > 0:
        lang = lang.replace("en-", "")
        logits_files = [os.path.join(args.pkl_dir, "{}.{}.logits.pkl{}".format(args.task_name, lang, idx))
                        for idx in args.pkl_index.split(',')]
        assert len(logits_files) > 0
        logger.info("Read logits files: {}".format(logits_files))
        soft_labels = np.zeros((len(features), args.max_seq_length, len(labels)))
        for f in logits_files:
            pkl_data = pickle.load(open(f, 'rb'))
            assert len(pkl_data) == len(features)
            soft_labels += pkl_data
        soft_labels /= len(logits_files)

    if use_barrier and args.local_rank == 0 and mode == 'train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    if mode == 'train' and args.alpha > 0:
        all_soft_labels = torch.from_numpy(soft_labels)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_soft_labels)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

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
        # for each aligned sample, choose from one target language
        dataset_idx = [random.choice(range(len(self.datasets))) for _ in range(self.dataset_len)]
        return [ dataset_idx[i]*self.dataset_len + i for i in range(self.dataset_len)]

    def shuffle(self):
        self.sample_idx = self.get_sample_idx()

def save_predictions(args, predictions, output_file, text_file, idx_file, output_word_prediction=False):
  # Save predictions
  with open(text_file, "r") as text_reader, open(idx_file, "r") as idx_reader:
    text = text_reader.readlines()
    index = idx_reader.readlines()
    assert len(text) == len(index)

  # Sanity check on the predictions
  with open(output_file, "w") as writer:
    example_id = 0
    prev_id = int(index[0])
    for line, idx in zip(text, index):
      if line == "" or line == "\n":
        example_id += 1
      else:
        cur_id = int(idx)
        output_line = '\n' if cur_id != prev_id else ''
        if output_word_prediction:
          output_line += line.split()[0] + '\t'
        output_line += predictions[example_id].pop(0) + '\n'
        writer.write(output_line)
        prev_id = cur_id


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
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
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument("--filter_k", type=int, default=0, help="the number fusing layers in FILTER")
    parser.add_argument("--filter_m", type=int, default=0, help="the number local transformers layers in FILTER")
    parser.add_argument("--alpha", type=float, default=0, help='alpha for self-teaching loss')
    parser.add_argument("--multiplier", type=float, default=1, help='multiplier for KD loss')
    parser.add_argument("--temperature", type=float, default=1, help="temprature to soft logits")

    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="The output log dir."
    )

    # Other parameters
    parser.add_argument(
        "--benchmark", default='xtreme', type=str, choices=['xtreme', 'xglue'], help="benchmark, xglue/xtreme")
    parser.add_argument(
        "--recover_model", type=boolean_string, default='true', help="recover model from checkpoint")
    parser.add_argument(
        "--single_line", action="store_true", help="Set this flag if training data is as single line format"
    )
    parser.add_argument(
        "--language",
        default=None,
        type=str,
        required=True,
        help="Evaluation language. Also train language if `train_language` is set to None.",
    )
    parser.add_argument(
        "--train_language", default=None, type=str, help="Train language if is different of the evaluation language."
    )
    parser.add_argument(
        "--eval_splits", default='dev', type=str, help="which splits to evaluate"
    )
    parser.add_argument("--eval_train", default='false', type=boolean_string, help="evaluate training set for KD")
    parser.add_argument("--pkl_index", default="0", type=str, help="use for dump training logits")
    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--gpu_id", default="", type=str, help="GPU id"
    )

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
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

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--keep_accents", action="store_const", const=True, help="Set this flag if model is trained with accents."
    )
    parser.add_argument(
        "--strip_accents", action="store_const", const=True, help="Set this flag if model is trained without accents."
    )
    parser.add_argument("--use_fast", action="store_const", const=True, help="Set this flag to use fast tokenization.")
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
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--use_all_samples_per_epoch", action='store_true', help="Use all samples for per epoch training"
    )

    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--logging_each_epoch", action="store_true", help="Logging every epoch")

    parser.add_argument(
        "--task_name",
        default="ner",
        type=str,
        required=True,
        help="The name of the task to train",
    )

    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=62, help="random seed for initialization")

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

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
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

    # Set seed
    set_seed(args)

    # Prepare CONLL-2003 task
    labels_list = get_labels(args.labels)
    num_labels = len(labels_list)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(labels_list)},
        label2id={label: i for i, label in enumerate(labels_list)},
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.filter_k > 0: # fusing layer exists, otherwise use default XLM
        config.alpha = args.alpha
        config.multiplier = args.multiplier
        config.temperature = args.temperature
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.output_hidden_states = True

        config.filter_m = args.filter_m
        config.filter_k = min(args.filter_k, config.num_hidden_layers - args.filter_m)
        config.num_hidden_layers = args.filter_m

    tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
    logger.info("Tokenizer arguments: %s", tokenizer_args)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        **tokenizer_args,
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

        dataset_list, exist_train_langs = [], []

        if args.local_rank not in [-1, 0]: # Make sure only first process create examples
            torch.distributed.barrier() 

        if args.train_language == 'all' or args.filter_k > 0: # use target languages for FILTER
            train_langs = ["en-{}".format(lang) if lang != 'en' else lang for lang in args.language.split(',')]
        else:
            train_langs = args.train_language.split(',')

        for lang in train_langs:
            # TODO: exclude some languages which don't have space for POS now as the prediction is word level
            if args.task_name in ['udpos', 'pos'] and lang in ['en-zh', 'en-ja', 'en-th']:
                continue
            lg_train_dataset = load_and_cache_examples(args,
                                                       tokenizer,
                                                       labels_list,
                                                       pad_token_label_id,
                                                       mode="train",
                                                       lang=lang,
                                                       use_barrier=False)
            if lg_train_dataset is not None:
                dataset_list.append(lg_train_dataset)
                exist_train_langs.append(lang)

        if args.filter_k > 0:
            # align dataset across languages
            train_dataset = AlignDataset(dataset_list,
                                         exist_train_langs.index('en'),
                                         is_training=True,
                                         use_all_samples=args.use_all_samples_per_epoch)
        else:
            train_dataset = ConcatDataset(dataset_list)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels_list, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_eval and args.local_rank in [-1, 0]:
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
        measure_metric = 'f1'
        if args.benchmark == 'xglue' and args.task_name == 'pos':
            measure_metric = 'acc'

        log_writer = open(os.path.join(args.output_dir, "eval_logs.txt"), 'w')
        for checkpoint in checkpoints:
            tokenizer = tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)


            results, predictions = evaluate(args, model, tokenizer, labels_list, pad_token_label_id, eval_splits=args.eval_splits.split(','))
            logger.info(results)
            log_writer.write("{}\t{}\n".format(checkpoint, results))

            cur_avg = results['dev_avg'][measure_metric]
            if cur_avg > best_avg:
                best_avg = cur_avg
                best_checkpoint = checkpoint

            # Save predictions
            for key, pred_list in predictions.items():
                split, lang = key.split('-')
                basename = os.path.basename(checkpoint)
                prefix = basename + "_" if basename else ""
                output_test_predictions_file = os.path.join(args.output_dir, "{}{}_{}_predictions.txt".format(prefix, split, lang))

                if args.benchmark == 'xtreme':
                    infile = os.path.join(args.data_dir, lang, "{}.{}".format(split, os.path.basename(args.model_name_or_path)))
                    idxfile = infile + '.idx'
                    save_predictions(args, pred_list, output_test_predictions_file, infile, idxfile)
                else:
                    with open(output_test_predictions_file, "w") as writer:
                        for sentence in pred_list:
                            for word in sentence:
                                writer.write(word + "\n")
                        writer.write("\n")

        logging.info("Best checkpoint: {}".format(best_checkpoint))
        log_writer.close()

    if args.eval_train and args.local_rank in [-1, 0]:
        if args.eval_checkpoints:
            # use the first one
            checkpoint = [os.path.join(args.output_dir, ckpt) for ckpt in args.eval_checkpoints.split(',')][0]
        else:
            checkpoint = args.output_dir
        assert os.path.exists(checkpoint)
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        evaluate(args, model, tokenizer, labels_list, pad_token_label_id, eval_splits=['train'])

if __name__ == "__main__":
    main()
