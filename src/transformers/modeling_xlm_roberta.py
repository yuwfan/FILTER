# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
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
"""PyTorch XLM-RoBERTa model. """


import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, KLDivLoss
import torch.nn.functional as F

from .configuration_xlm_roberta import XLMRobertaConfig
from .configuration_roberta import RobertaConfig
from .file_utils import add_start_docstrings
from .modeling_roberta import (
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForSequenceClassification,
    RobertaForMultiTaskSequenceClassification,
    RobertaForTokenClassification,
    RobertaForQuestionAnswering,
    RobertaModel,
    RobertaClassificationHead,
)
from .modeling_bert import BertPreTrainedModel, BertAttention, BertLayer

logger = logging.getLogger(__name__)

XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "xlm-roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-base-pytorch_model.bin",
    "xlm-roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-dutch": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-dutch-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-spanish": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-spanish-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-english": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-english-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-german": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-german-pytorch_model.bin",
}


XLM_ROBERTA_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.XLMRobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""


@add_start_docstrings(
    "The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaModel(RobertaModel):
    """
    This class overrides :class:`~transformers.RobertaModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """XLM-RoBERTa Model with a `language modeling` head on top. """, XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForMaskedLM(RobertaForMaskedLM):
    """
    This class overrides :class:`~transformers.RobertaForMaskedLM`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

@add_start_docstrings(
    """XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForMultiTaskSequenceClassification(RobertaForMultiTaskSequenceClassification):
    """
    This class overrides :class:`~transformers.RobertaForSequenceClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

class FilterModel(RobertaModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config):
        super().__init__(config)

        #self.local = RobertaModel(config)
        self.filter_m = config.filter_m
        self.filter_k = config.filter_k

        # cross attention
        if self.filter_k:
            assert self.filter_k >= 1
            # TODO: hack for now, initialize in modeling_utils
            self.fusion = nn.ModuleList([BertLayer(config) for _ in range(self.filter_k)])

            # remaining encoder layer, hack for now
            layer_num = 24 - (self.filter_m + config.filter_k)
            self.domain = nn.ModuleList([BertLayer(config) for _ in range(layer_num)])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        soft_labels=None,
    ):
        def extend_attention_mask(mask):
            if mask.dim() == 2:
                extended_mask = mask[:, None, None, :]
                extended_mask = extended_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_mask = (1.0 - extended_mask) * -10000.0
            return extended_mask


        assert isinstance(input_ids, list)
        assert self.filter_k > 0
        all_outputs = []
        cls_index, cls_indexes = 0, []
        for i in range(len(input_ids)):
            #outputs = self.roberta(input_ids[i],
            outputs = super().forward(input_ids[i],
                                      attention_mask=attention_mask[i] if attention_mask is not None else None,
                                      token_type_ids=token_type_ids[i] if token_type_ids is not None else None,
                                      position_ids=position_ids[i] if position_ids is not None else None,
                                      head_mask=head_mask[i] if head_mask is not None else None, 
                                      inputs_embeds=inputs_embeds[i] if inputs_embeds is not None else None)

            all_outputs.append(outputs[2][self.filter_m])

            cls_indexes.append(cls_index)
            cls_index += input_ids[i].size(1)
        extended_attention_mask = None
        if attention_mask:
            extended_attention_mask = extend_attention_mask(torch.cat(attention_mask, dim=1))
        hidden_states = torch.cat(all_outputs, dim=1)
        for i, layer_module in enumerate(self.fusion):
            layer_outputs = layer_module(
                hidden_states,
                extended_attention_mask,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None)
            hidden_states = layer_outputs[0]
        sequence_output = hidden_states
        del all_outputs

        if len(self.domain) > 0:
            mix_sequence_output = []
            for i in range(len(cls_indexes)):
                if i == len(cls_indexes)-1:
                    hidden_states = sequence_output[:, cls_indexes[i]:, :]
                else:
                    hidden_states = sequence_output[:, cls_indexes[i]:cls_indexes[i+1], :]

                extended_attention_mask = extend_attention_mask(attention_mask[i])
                for i, layer_module in enumerate(self.domain):
                    layer_outputs = layer_module(
                        hidden_states,
                        extended_attention_mask if attention_mask is not None else None, 
                        head_mask=None,
                        encoder_hidden_states=None,
                        encoder_attention_mask=None)
                    hidden_states = layer_outputs[0]
                mix_sequence_output.append(hidden_states)
            sequence_output = torch.cat(mix_sequence_output, dim=1)
        return sequence_output



@add_start_docstrings(
    """XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    """
    This class overrides :class:`~transformers.RobertaForSequenceClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

@add_start_docstrings(
    """XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """,
    XLM_ROBERTA_START_DOCSTRING,
)
class FilterForSequenceClassification(BertPreTrainedModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config): 
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.filter = FilterModel(config)

        self.output = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, config.num_labels))


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        soft_labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
        """
        outputs = self.filter(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        logits = self.output(outputs)

        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                other_lang_logits, eng_logits = logits[:, 0, :].view(-1, self.num_labels), logits[:, 1, :].view(-1, self.num_labels)

                loss_other = loss_fct(other_lang_logits, labels.view(-1))
                loss_eng = loss_fct(eng_logits, labels.view(-1))

                if self.config.alpha > 0:
                    alpha = self.config.alpha
                    T = self.config.temperature
                    loss_KD = KLDivLoss()(F.log_softmax(other_lang_logits/T, dim=1), F.softmax(soft_labels/T, dim=1)) * (T * T)
                    loss =  (1. - alpha) * loss_other + alpha * loss_KD
                else:
                    loss = loss_other

                if not self.config.first_loss_only:
                    loss += loss_eng
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

@add_start_docstrings(
    """XLM-RoBERTa Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForMultipleChoice(RobertaForMultipleChoice):
    """
    This class overrides :class:`~transformers.RobertaForMultipleChoice`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """XLM-RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    """
    This class overrides :class:`~transformers.RobertaForQuestionAnswering`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


@add_start_docstrings(
    """XLM-RoBERTa Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForTokenClassification(RobertaForTokenClassification):
    """
    This class overrides :class:`~transformers.RobertaForTokenClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


class FilterForTokenClassification(BertPreTrainedModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.filter = FilterModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.alpha = self.config.alpha
        self.T = self.config.temperature
        self.kd_loss_fct = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        soft_labels=None,
    ):
        outputs = self.filter(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = self.dropout(outputs)
        logits = self.classifier(sequence_output)

        logits_list = logits.split([input_id.size(1) for input_id in input_ids], dim=1)
        logits, eng_logits = logits_list[0].contiguous(), logits_list[1].contiguous()
        outputs = (logits,)

        if labels is not None:
            def calc_loss(attention_mask, logits, labels):
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss

            eng_attention_mask=attention_mask[1] if attention_mask is not None else None
            loss_eng = calc_loss(eng_attention_mask, eng_logits, labels[1])

            if self.alpha > 0:
                cur_attention_mask=attention_mask[0] if attention_mask is not None else None
                loss_self = calc_loss(cur_attention_mask, logits, labels[0])

                mask = cur_attention_mask.unsqueeze(-1).expand_as(logits)  # (bs, seq_lenth, label_size)
                s_logits_sel = torch.masked_select(logits, mask == 1)
                s_logits_sel = s_logits_sel.view(-1, logits.size(-1))
                t_logits_sel = torch.masked_select(soft_labels[0], mask==1)
                t_logits_sel = t_logits_sel.view(-1, logits.size(-1))
                assert t_logits_sel.size() == s_logits_sel.size()

                loss_KD = self.kd_loss_fct(F.log_softmax(s_logits_sel/self.T, dim=-1), F.softmax(t_logits_sel/self.T, dim=-1)) * (self.T)**2

                loss =  (1. - self.alpha) * loss_self + self.alpha * loss_KD + loss_eng
            else:
                loss = loss_eng

            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)



@add_start_docstrings(
    """XLM-Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    XLM_ROBERTA_START_DOCSTRING,
)
class FilterForQuestionAnswering(BertPreTrainedModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.filter = FilterModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        #self.init_weights()
        self.kd_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.alpha = self.config.alpha
        self.T = self.config.temperature

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        start_soft_positions=None,
        end_soft_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        # The checkpoint roberta-large is not fine-tuned for question answering. Please see the
        # examples/run_squad.py example to see how to fine-tune a model to a question answering task.

        from transformers import RobertaTokenizer, RobertaForQuestionAnswering
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForQuestionAnswering.from_pretrained('roberta-base')

        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        start_scores, end_scores = model(torch.tensor([input_ids]))

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

        """
        outputs = self.filter(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        logits = self.qa_outputs(outputs)

        logits_list = logits.split([input_id.size(1) for input_id in input_ids], dim=1)
        start_logits, end_logits = [], []
        for i in range(len(logits_list)):
            sl, el = logits_list[i].split(1, dim=-1)
            start_logits.append(sl.squeeze(-1))
            end_logits.append(el.squeeze(-1))

        outputs = (start_logits[0], end_logits[0],)

        if start_positions is not None and end_positions is not None:
            def calc_loss(sp, ep, sl, el):
                if len(sp.size()) > 1:
                    sp = sp.squeeze(-1)
                if len(ep.size()) > 1:
                    ep = ep.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = sl.size(1)
                sp.clamp_(0, ignored_index)
                ep.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(sl, sp)
                end_loss = loss_fct(el, ep)
                total_loss = (start_loss + end_loss) / 2

                return total_loss

            def calc_masked_kd_loss(mask, sp, ep, sl, el):
                sl_sel = torch.masked_select(sl, mask == 1)
                el_sel = torch.masked_select(el, mask == 1)
                sp_sel = torch.masked_select(sp, mask == 1)
                ep_sel = torch.masked_select(ep, mask == 1)

                s_kd_loss = self.kd_loss_fct(F.log_softmax(sl_sel/self.T, dim=-1), F.softmax(sp_sel/self.T, dim=-1)) * (self.T * self.T)
                e_kd_loss = self.kd_loss_fct(F.log_softmax(el_sel/self.T, dim=-1), F.softmax(ep_sel/self.T, dim=-1)) * (self.T * self.T)

                return (s_kd_loss + e_kd_loss) / 2


            if isinstance(start_positions, list):
                total_loss = 0

                loss_other = calc_loss(start_positions[0], end_positions[0], start_logits[0], end_logits[0])
                loss_eng = calc_loss(start_positions[1], end_positions[1], start_logits[1], end_logits[1]) if not self.config.first_loss_only else 0

                if self.alpha > 0:

                    # select start_logits, and end_logits
                    kd_loss = calc_masked_kd_loss(attention_mask[0], start_soft_positions[0],
                                                  end_soft_positions[0], start_logits[0], end_logits[0])

                    total_loss =  (1. - self.alpha) * loss_other + self.alpha * kd_loss + loss_eng
                    #loss_start_KD = KLDivLoss()(F.log_softmax(start_logits[0]/T, dim=1), F.softmax(start_soft_positions[0]/T, dim=1)) * (T * T)
                    #loss_end_KD = KLDivLoss()(F.log_softmax(end_logits[0]/T, dim=1), F.softmax(end_soft_positions[0]/T, dim=1)) * (T * T)
                    #total_loss =  (1. - alpha) * loss_other + alpha * ((loss_start_KD+loss_end_KD)/2) + loss_eng
                else:
                    total_loss = loss_other + loss_eng
            else:
                total_loss = calc_loss(start_positions, end_positions, start_logits, end_logits)
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
