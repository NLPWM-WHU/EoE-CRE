import copy
import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler


class BaseData:
    def __init__(self, args):
        self.args = args
        self.label_list = self._read_labels()
        self.id2label, self.label2id = [], {}
        self.label2task_id = {}
        self.train_data, self.val_data, self.test_data = None, None, None

    def _read_labels(self):
        """
        :return: only return the label name, in order to set label index from 0 more conveniently.
        """
        id2label = json.load(open(os.path.join(self.args.data_path, self.args.dataset_name, 'id2label.json')))
        return id2label

    def read_and_preprocess(self, **kwargs):
        raise NotImplementedError

    def add_labels(self, cur_labels, task_id):
        for c in cur_labels:
            if c not in self.id2label:
                self.id2label.append(c)
                self.label2id[c] = len(self.label2id)
                self.label2task_id[self.label2id[c]] = task_id

    def filter(self, labels, split='train'):
        if not isinstance(labels, list):
            labels = [labels]
        split = split.lower()
        res = []
        for label in labels:
            if split == 'train':
                if self.args.debug:
                    res += copy.deepcopy(self.train_data[label])[:10]
                else:
                    res += copy.deepcopy(self.train_data[label])
            elif split in ['dev', 'val']:
                if self.args.debug:
                    res += copy.deepcopy(self.val_data[label])[:10]
                else:
                    res += copy.deepcopy(self.val_data[label])
            elif split == 'test':
                if self.args.debug:
                    res += copy.deepcopy(self.test_data[label])[:10]
                else:
                    res += copy.deepcopy(self.test_data[label])
        for idx in range(len(res)):
            res[idx]["labels"] = self.label2id[res[idx]["labels"]]
        return res


class BaseDataset(Dataset):
    def __init__(self, data):
        if isinstance(data, dict):
            res = []
            for key in data.keys():
                res += data[key]
            data = res
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # cur_data = self.data[idx]
        # cur_data["idx"] = idx
        # mask_head = True if random.random() > 0.5 else False
        # input_ids, attention_mask, subject_start_pos, object_start_pos = mask_entity(cur_data["input_ids"], mask_head)
        # augment_data = {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "subject_start_pos": subject_start_pos,
        #     "object_start_pos": object_start_pos,
        #     "labels": cur_data["labels"],
        #     "idx": idx
        # }
        # return [cur_data, augment_data]
        return self.data[idx]

