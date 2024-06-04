import copy
import os
from typing import List, Dict, Any, Optional, Union

import torch
from attr import dataclass
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, set_seed, PreTrainedTokenizerBase
from torch.optim import AdamW
from transformers.utils import PaddingStrategy

from data import BaseDataset
import logging
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn import metrics
from sklearn import manifold
from utils import relation_data_augmentation, CustomCollatorWithPadding

logger = logging.getLogger(__name__)


class ExpertTrainer:
    def __init__(self, args, **kwargs):
        self.optimizer = None
        self.task_idx = 0
        self.args = args

    def run(self, data, model, tokenizer, label_order, seed=None):
        if seed is not None:
            set_seed(seed)
        default_data_collator = CustomCollatorWithPadding(tokenizer)

        seen_labels = []
        all_cur_acc = [0] * self.args.num_tasks
        all_total_acc = [0] * self.args.num_tasks
        all_total_hit = [0] * self.args.num_tasks
        marker_ids = tuple([tokenizer.convert_tokens_to_ids(c) for c in self.args.additional_special_tokens])
        logger.info(f"marker ids: {marker_ids}")
        for task_idx in range(self.args.num_tasks):
            self.task_idx = task_idx
            cur_labels = [data.label_list[c] for c in label_order[task_idx]]
            data.add_labels(cur_labels, task_idx)
            seen_labels += cur_labels

            logger.info(f"***** Task-{task_idx + 1} *****")
            logger.info(f"Current classes: {' '.join(cur_labels)}")

            train_data = data.filter(cur_labels, "train")
            # data augmentation
            num_train_labels = len(cur_labels)
            train_data, num_train_labels = relation_data_augmentation(
                train_data, len(seen_labels), copy.deepcopy(data.id2label), marker_ids, self.args.augment_type
            )
            train_dataset = BaseDataset(train_data)

            model.new_task(num_train_labels)

            self.train(
                model=model,
                train_dataset=train_dataset,
                data_collator=default_data_collator
            )
            cur_test_data = data.filter(cur_labels, 'test')
            cur_test_dataset = BaseDataset(cur_test_data)
            cur_result = self.eval(
                model=model,
                eval_dataset=cur_test_dataset,
                data_collator=default_data_collator,
                seen_labels=seen_labels,
            )

            os.makedirs(self.args.save_model_dir, exist_ok=True)
            save_model_name = f"{self.args.dataset_name}_{seed}_{self.args.augment_type}.pth"
            save_model_path = os.path.join(self.args.save_model_dir, save_model_name)
            logger.info(f"save expert model to {save_model_path}")
            self.save_model(model, save_model_path)

            all_cur_acc[self.task_idx] = cur_result
            all_total_acc[self.task_idx] = cur_result
            all_total_hit[self.task_idx] = 1
            # only for the first task
            if self.task_idx == 0:
                break

        return {
            "cur_acc": all_cur_acc,
            "total_acc": all_total_acc,
            "total_hit": all_total_hit,
        }

    def train(self, model, train_dataset, data_collator):
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=data_collator
        )
        len_dataloader = len(train_dataloader)
        num_examples = len(train_dataset)
        max_steps = len_dataloader * self.args.num_train_epochs

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Train batch size = {self.args.train_batch_size}")
        logger.info(f"  Total optimization steps = {max_steps}")

        no_decay = ["bias", "LayerNorm.weight"]
        parameters = [
            {'params': [p for n, p in model.named_parameters() if 'feature_extractor' in n and not any(nd in n for nd in no_decay)],
             'lr': self.args.learning_rate, 'weight_decay': 1e-2},
            {'params': [p for n, p in model.named_parameters() if 'feature_extractor' in n and any(nd in n for nd in no_decay)],
             'lr': self.args.learning_rate, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if 'feature_extractor' not in n and not any(nd in n for nd in no_decay)],
             'lr': self.args.classifier_learning_rate, 'weight_decay': 1e-2},
            {'params': [p for n, p in model.named_parameters() if 'feature_extractor' not in n and any(nd in n for nd in no_decay)],
             'lr': self.args.classifier_learning_rate, 'weight_decay': 0.0},
        ]
        self.optimizer = AdamW(parameters)

        progress_bar = tqdm(range(max_steps))

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        for epoch in range(self.args.num_train_epochs):
            model.train()
            for step, inputs in enumerate(train_dataloader):
                self.optimizer.zero_grad()

                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                self.optimizer.step()

                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": loss.item()})

        progress_bar.close()

    @torch.no_grad()
    def eval(self, model, eval_dataset, data_collator, seen_labels):
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

        len_dataloader = len(eval_dataloader)
        num_examples = len(eval_dataset)

        logger.info("***** Running evaluating *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Eval batch size = {self.args.eval_batch_size}")

        progress_bar = tqdm(range(len_dataloader))

        golds = []
        preds = []

        model.eval()
        for step, inputs in enumerate(eval_dataloader):
            labels = inputs.pop('labels')
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}

            outputs = model(**inputs)

            logits = outputs.logits

            predicts = logits.max(dim=-1)[1]

            predicts = predicts.cpu().tolist()
            labels = labels.cpu().tolist()
            golds.extend(predicts)
            preds.extend(labels)

            progress_bar.update(1)
        progress_bar.close()

        micro_f1 = metrics.f1_score(golds, preds, average='micro')
        logger.info("Micro F1 {}".format(micro_f1))

        return micro_f1

    def save_model(self, model, save_path):
        bert_state_dict = model.feature_extractor.bert.state_dict()
        linear_state_dict = model.classifier.state_dict()
        torch.save({
            "model": bert_state_dict,
            "linear": linear_state_dict,
        }, save_path)

