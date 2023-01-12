# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Question-Answering task와 관련된 'Trainer'의 subclass 코드 입니다.
"""
from pooler import BertPooler

from transformers import Trainer, is_datasets_available, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput
from typing import  Dict, Optional
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

import torch
from torch import nn

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

def js_div( p, q):
    m = (p + q) / 2
    a = F.kl_div(p.log(), m, reduction='batchmean')
    b = F.kl_div(q.log(), m, reduction='batchmean')
    jsd = ((a + b) / 2)
    return jsd

# Huggingface의 Trainer를 상속받아 QuestionAnswering을 위한 Trainer를 생성합니다.
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.config = self.model.config
        self.encoder = self.model.roberta.encoder
        self.pooler = BertPooler(self.config)
        self.qa_outputs = self.model.qa_outputs


    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # 일시적으로 metric computation를 불가능하게 한 상태이며, 해당 코드에서는 loop 내에서 metric 계산을 수행합니다.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions, self.args
            )
            metrics = self.compute_metrics(eval_preds)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: PyTorch/XLA에 대한 Logging debug metrics (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)

        # 일시적으로 metric computation를 불가능하게 한 상태이며, 해당 코드에서는 loop 내에서 metric 계산을 수행합니다.
        # evaluate 함수와 동일하게 구성되어있습니다
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"],
                columns=list(test_dataset.features.keys()),
            )

        predictions = self.post_process_function(
            test_examples, test_dataset, output.predictions, self.args
        )
        return predictions
         
    def get_bert_output(self, embedding_output, attention_mask=None):
        # input_shape = embedding_output.size().tolist()[:2]
        # device = embedding_output.device

        assert not self.config.is_decoder
        assert attention_mask.dim() == 2

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.model.roberta.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)


        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

         
    def get_logits_from_embedding_output(self, embedding_output, attention_mask=None, start_positions=None, end_positions=None):

        outputs = self.get_bert_output(embedding_output, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output

        # outputs = (logits,) + outputs[2:]
        # print('logits: ',logits)
        # if labels is not None:
        #     if self.num_labels == 1:
        #         #  We are doing regression
        #         loss_fct = MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     outputs = (loss,) + outputs
        # return outputs  # (loss), logits, (hidden_states), (attentions)

    def generate_token_cutoff_embedding(self, embeds, masks, input_lens):
        input_embeds = []
        input_masks = []
        for i in range(embeds.shape[0]):
            cutoff_length = int(input_lens[i] * self.args.aug_cutoff_ratio)
            zero_index = torch.randint(input_lens[i], (cutoff_length,))

            cutoff_embed = embeds[i]
            cutoff_mask = masks[i]

            tmp_mask = torch.ones(cutoff_embed.shape[0], ).to(self.args.device)
            for ind in zero_index:
                tmp_mask[ind] = 0

            cutoff_embed = torch.mul(tmp_mask[:, None], cutoff_embed)
            cutoff_mask = torch.mul(tmp_mask, cutoff_mask).type(torch.int64)

            input_embeds.append(cutoff_embed)
            input_masks.append(cutoff_mask)

        input_embeds = torch.stack(input_embeds, dim=0)
        input_masks = torch.stack(input_masks, dim=0)

        return input_embeds, input_masks

    def generate_dim_cutoff_embedding(self, embeds, masks, input_lens):
        input_embeds = []
        input_masks = []
        for i in range(embeds.shape[0]):
            cutoff_embed = embeds[i]
            cutoff_mask = masks[i]

            cutoff_length = int(cutoff_embed.shape[1] * self.args.aug_cutoff_ratio)
            zero_index = torch.randint(cutoff_embed.shape[1], (cutoff_length,))

            tmp_mask = torch.ones(cutoff_embed.shape[1], ).to(self.args.device)
            for ind in zero_index:
                tmp_mask[ind] = 0.

            cutoff_embed = torch.mul(tmp_mask, cutoff_embed)

            input_embeds.append(cutoff_embed)
            input_masks.append(cutoff_mask)
        input_embeds = torch.stack(input_embeds, dim=0)
        input_masks = torch.stack(input_masks, dim=0)

        return input_embeds, input_masks

    def _resolve_loss_item(self, loss):

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()
    
    def training_step_with_span_cutoff(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> float:
        # optimizer = self.optimizer #나중에 바꾸기
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        ori_outputs = model(**inputs)
        loss = ori_outputs[0]  # model outputs are always tuple in transformers (see doc)

        # assert model.__class__ is RobertaForSequenceClassification

        # Cut embedding_output and attention mask
        input_ids = inputs['input_ids']
        # print("------------------------->",input_ids)
        
        token_type_ids = inputs.get('token_type_ids', None)
        start_positions = inputs.get('start_positions', None)
        end_positions = inputs.get('end_positions', None)
        # print("token type id: ", token_type_ids)

        #input id 와 token_type_id 로 embedding 생성 [8, 384, 768]
        embeds = model.roberta.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        # print(embeds.shape)

        masks = inputs['attention_mask']
        input_lens = torch.sum(masks, dim=1) #384중에 1로 된 애들의 개수를 나타낸 길이 8의 배열
        input_embeds = []
        input_masks = []
        for i in range(embeds.shape[0]): #batch_size만큼 반복
            cutoff_length = int(input_lens[i] * self.args.aug_cutoff_ratio)
            start = int(torch.rand(1).cuda() * (input_lens[i] - cutoff_length))
            # print("range별로?", input_lens[i], cutoff_length, start)
            cutoff_embed = torch.cat((embeds[i][:start],
                                      torch.zeros([cutoff_length, embeds.shape[-1]],
                                                  dtype=torch.float).to(self.args.device),
                                      embeds[i][start + cutoff_length:]), dim=0)
            cutoff_mask = torch.cat((masks[i][:start],
                                     torch.zeros([cutoff_length], dtype=torch.long).to(self.args.device),
                                     masks[i][start + cutoff_length:]), dim=0)
            input_embeds.append(cutoff_embed)
            input_masks.append(cutoff_mask)
        input_embeds = torch.stack(input_embeds, dim=0)
        input_masks = torch.stack(input_masks, dim=0)

        cutoff_outputs = self.get_logits_from_embedding_output(embedding_output=input_embeds,
                                                                attention_mask=input_masks, start_positions = start_positions, end_positions=end_positions)
        if self.args.aug_ce_loss > 0:
            loss += self.args.aug_ce_loss * cutoff_outputs[0]

        if self.args.aug_js_loss > 0:
            assert self.args.n_gpu == 1
            ori_logits = ori_outputs[1]
            aug_logits = cutoff_outputs[1]
            p = torch.softmax(ori_logits + 1e-10, dim=1)
            q = torch.softmax(aug_logits + 1e-10, dim=1)
            aug_js_loss = js_div(p, q)
            loss += self.args.aug_js_loss * aug_js_loss

        return self._resolve_loss_item(loss)

    def training_step_with_token_cutoff(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor]
    ) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        ori_outputs = model(**inputs)
        #loss = ori_outputs[0]  # model outputs are always tuple in transformers (see doc)
        loss = 0.0

        # assert model.__class__ is RobertaForSequenceClassification
        # if self.args.aug_version == 'v3':
        input_ids = inputs['input_ids']
        token_type_ids = inputs.get('token_type_ids', None)
        start_positions = inputs.get('start_positions', None)
        end_positions = inputs.get('end_positions', None)
        embeds = model.roberta.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        masks = inputs['attention_mask']
        input_lens = torch.sum(masks, dim=1)

        input_embeds, input_masks = self.generate_token_cutoff_embedding(embeds, masks, input_lens)
        cutoff_outputs = self.get_logits_from_embedding_output(embedding_output=input_embeds,
                                                                attention_mask=input_masks, start_positions = start_positions, end_positions=end_positions)

        if self.args.aug_ce_loss > 0:
            loss += self.args.aug_ce_loss * cutoff_outputs[0]

        if self.args.aug_js_loss > 0:
            assert self.args.n_gpu == 1
            ori_logits = ori_outputs[1]
            aug_logits = cutoff_outputs[1]
            p = torch.softmax(ori_logits + 1e-10, dim=1)
            q = torch.softmax(aug_logits + 1e-10, dim=1)
            aug_js_loss = js_div(p, q)
            loss += self.args.aug_js_loss * aug_js_loss

        return self._resolve_loss_item(loss)

    def training_step_with_dim_cutoff(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        ori_outputs = model(**inputs)
        loss = ori_outputs[0]  # model outputs are always tuple in transformers (see doc)

        # assert model.__class__ is RobertaForSequenceClassification
        input_ids = inputs['input_ids']
        token_type_ids = inputs.get('token_type_ids', None)
        start_positions = inputs.get('start_positions', None)
        end_positions = inputs.get('end_positions', None)

        embeds = model.roberta.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        masks = inputs['attention_mask']
        input_lens = torch.sum(masks, dim=1)

        input_embeds, input_masks = self.generate_dim_cutoff_embedding(embeds, masks, input_lens)
        cutoff_outputs = self.get_logits_from_embedding_output(embedding_output=input_embeds,
                                                               attention_mask=input_masks,
                                                               start_positions=start_positions,
                                                               end_positions=end_positions)

        if self.args.aug_ce_loss > 0:
            loss += self.args.aug_ce_loss * cutoff_outputs[0]

        if self.args.aug_js_loss > 0:
            assert self.args.n_gpu == 1
            ori_logits = ori_outputs[1]
            aug_logits = cutoff_outputs[1]
            p = torch.softmax(ori_logits + 1e-10, dim=1)
            q = torch.softmax(aug_logits + 1e-10, dim=1)
            aug_js_loss = js_div(p, q)
            loss += self.args.aug_js_loss * aug_js_loss

        return self._resolve_loss_item(loss)