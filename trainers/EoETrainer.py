import copy
import logging
import os
import pickle

import hydra
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

from data import BaseDataset
from trainers import BaseTrainer
from utils import CustomCollatorWithPadding, relation_data_augmentation

logger = logging.getLogger(__name__)


class EoETrainer(BaseTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.task_idx = 0
        self.cur_seed = 0

    def run(self, data, model, tokenizer, label_order, seed=None):
        if seed is not None:
            set_seed(seed)
            self.cur_seed = seed
        default_data_collator = CustomCollatorWithPadding(tokenizer)

        seen_labels = []
        all_cur_acc = []
        all_total_acc = []
        all_total_hit = []
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
            train_dataset = BaseDataset(train_data)
            num_train_labels = len(cur_labels)
            aug_train_data, num_train_labels = relation_data_augmentation(
                copy.deepcopy(train_data), len(seen_labels), copy.deepcopy(data.id2label), marker_ids, self.args.augment_type
            )
            aug_train_dataset = BaseDataset(aug_train_data)
            model.new_task(num_train_labels)

            if self.task_idx == 0:
                expert_model = f"./ckpt/{self.args.dataset_name}_{seed}_{self.args.augment_type}.pth"
                model.load_expert_model(expert_model)
                logger.info(f"load first task model from {expert_model}")
            else:
                self.train(
                    model=model,
                    train_dataset=aug_train_dataset,
                    data_collator=default_data_collator
                )

            os.makedirs(f"./ckpt/{self.args.dataset_name}-{seed}-{self.args.augment_type}", exist_ok=True)
            model.save_classifier(
                idx=self.task_idx,
                save_dir=f"./ckpt/{self.args.dataset_name}-{seed}-{self.args.augment_type}",
            )

            model.feature_extractor.save_and_load_all_adapters(
                self.task_idx,
                save_dir=f"./ckpt/{self.args.dataset_name}-{seed}-{self.args.augment_type}",
                save=True,
            )

            self.statistic(model, train_dataset, default_data_collator)

            cur_test_data = data.filter(cur_labels, 'test')
            history_test_data = data.filter(seen_labels, 'test')

            cur_test_dataset = BaseDataset(cur_test_data)
            history_test_dataset = BaseDataset(history_test_data)

            cur_acc, cur_hit = self.eval(
                model=model,
                eval_dataset=cur_test_dataset,
                data_collator=default_data_collator,
                seen_labels=seen_labels,
                label2task_id=copy.deepcopy(data.label2task_id),
                oracle=True,
            )

            total_acc, total_hit = self.eval(
                model=model,
                eval_dataset=history_test_dataset,
                data_collator=default_data_collator,
                seen_labels=seen_labels,
                label2task_id=copy.deepcopy(data.label2task_id),
            )

            all_cur_acc.append(cur_acc)
            all_total_acc.append(total_acc)
            all_total_hit.append(total_hit)

        # save distribution
        save_data = {
            "distribution": model.expert_distribution,
            "seen_labels": seen_labels,
            "label2id": data.label2id,
        }
        save_file = f"{self.cur_seed}_distribution.pickle"
        save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        with open(save_dir + "/" + save_file, 'wb') as file:
            pickle.dump(save_data, file)

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
        for name, param in model.named_parameters():
            if param.requires_grad and "lora_" in name:
                print(name)
                break

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
    def eval(self, model, eval_dataset, data_collator, seen_labels, label2task_id, oracle=False):
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
        pred_indices = []
        gold_indices = []
        expert_task_preds = []
        expert_class_preds = []
        hits = 0
        model.eval()
        for step, inputs in enumerate(eval_dataloader):

            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            if oracle:
                inputs.update({"oracle": True, "task_idx": self.task_idx})
            outputs = model(**inputs)

            hit_pred = outputs.indices
            hit_gold = [label2task_id[c] for c in inputs["labels"].tolist()]
            pred_indices.extend(hit_pred)
            gold_indices.extend(hit_gold)

            predicts = outputs.preds.tolist()
            labels = inputs["labels"].tolist()
            golds.extend(labels)
            preds.extend(predicts)

            expert_task_preds.append(outputs.expert_task_preds)
            expert_class_preds.append(outputs.expert_class_preds)

            progress_bar.update(1)
        progress_bar.close()

        logger.info("\n" + metrics.classification_report(golds, preds))
        acc = metrics.accuracy_score(golds, preds)
        hit_acc = metrics.accuracy_score(gold_indices, pred_indices)
        logger.info("Acc {}".format(acc))
        logger.info("Hit Acc {}".format(hit_acc))

        if not oracle:
            expert_task_preds = torch.cat(expert_task_preds, dim=0).tolist()
            expert_class_preds = torch.cat(expert_class_preds, dim=0).tolist()
            save_data = {
                "preds": preds,
                "golds": golds,
                "pred_indices": pred_indices,
                "gold_indices": gold_indices,
                "expert_task_preds": expert_task_preds,
                "expert_class_preds": expert_class_preds,
            }
            # save information
            save_file = f"{self.cur_seed}_{self.task_idx}.pickle"
            save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            with open(save_dir + "/" + save_file, 'wb') as file:
                pickle.dump(save_data, file)

        return acc, hit_acc

    def statistic(self, model, dataset, data_collator):
        for i in range(-1, self.task_idx + 1):
            mean, cov, task_mean, task_cov = self.get_mean_and_cov(model, dataset, data_collator, i)
            model.new_statistic(mean, cov, task_mean, task_cov, i)

    @torch.no_grad()
    def get_mean_and_cov(self, model, dataset, data_collator, expert_id=0):
        loader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )
        model.eval()

        prelogits = []
        labels = []

        for step, inputs in enumerate(loader):
            label = inputs.pop('labels')
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            inputs.update({"return_hidden_states": True})
            inputs.update({"task_idx": expert_id})

            prelogit = model(**inputs)

            prelogits.extend(prelogit.tolist())
            labels.extend(label.tolist())

        prelogits = torch.tensor(prelogits)
        labels = torch.tensor(labels)
        labels_space = torch.unique(labels)

        task_mean = prelogits.mean(dim=0)
        task_cov = torch.cov((prelogits - task_mean).T)

        mean_over_classes = []
        cov_over_classes = []
        for c in labels_space:
            embeds = prelogits[labels == c]
            if embeds.numel() > 0:
                mean = embeds.mean(dim=0)
                cov = torch.cov((embeds - mean).T)
            else:
                mean = task_mean
                cov = task_cov
            mean_over_classes.append(mean)
            cov_over_classes.append(cov)

        mean_over_classes = torch.stack(mean_over_classes)
        shared_cov = torch.stack(cov_over_classes).mean(dim=0)

        return mean_over_classes, shared_cov, task_mean, task_cov
