import copy
import os
import json
from tqdm import tqdm
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

from .BaseData import BaseData


class TACREDData(BaseData):
    def __init__(self, args):
        super().__init__(args)
        self.entity_markers = ["[E11]", "[E12]", "[E21]", "[E22]"]
        # self.pretrain_re = self.args.pretrain_re

    def remove_entity_markers(self, input_ids):
        ans = []
        entity_pos = {}
        for c in input_ids:
            if c not in [30522, 30523, 30524, 30525]:
                ans.append(c)
            else:
                if c % 2 == 0:
                    entity_pos[c] = len(ans)
                else:
                    entity_pos[c] = len(ans) - 1
        return ans, entity_pos[30522], entity_pos[30523], entity_pos[30524], entity_pos[30525]

    def preprocess(self, raw_data, tokenizer):
        subject_start_marker = tokenizer.convert_tokens_to_ids(self.entity_markers[0])
        object_start_marker = tokenizer.convert_tokens_to_ids(self.entity_markers[2])
        subject_end_marker = tokenizer.convert_tokens_to_ids(self.entity_markers[1])
        object_end_marker = tokenizer.convert_tokens_to_ids(self.entity_markers[3])
        res = []
        result = tokenizer(raw_data['sentence'])
        for idx in range(len(raw_data['sentence'])):
            subject_marker_st = result['input_ids'][idx].index(subject_start_marker)
            object_marker_st = result['input_ids'][idx].index(object_start_marker)
            subject_marker_ed = result['input_ids'][idx].index(subject_end_marker)
            object_marker_ed = result['input_ids'][idx].index(object_end_marker)
            input_ids = result['input_ids'][idx]
            sentence = copy.deepcopy(raw_data['sentence'][idx])
            subject = sentence[sentence.index("[E11]") + 5: sentence.index("[E12]")].strip()
            object = sentence[sentence.index("[E21]") + 5: sentence.index("[E22]")].strip()
            for c in self.entity_markers:
                sentence = sentence.replace(c, '')
            sentence = sentence.replace('  ', ' ')
            input_ids_without_marker, subject_st, subject_ed, object_st, object_ed = \
                self.remove_entity_markers(input_ids)
            ins = {
                'sentence': sentence,
                'input_ids': input_ids,  # default: add marker to the head entity and tail entity
                'subject_marker_st': subject_marker_st,
                'object_marker_st': object_marker_st,
                'labels': raw_data['labels'][idx],
                'input_ids_without_marker': input_ids_without_marker,
                'subject_st': subject_st,
                'subject_ed': subject_ed,
                'object_st': object_st,
                'object_ed': object_ed,
                'subject': subject,
                'object': object,
            }
            if hasattr(self.args, 'columns'):
                columns = self.args.columns
                ins = {k: v for k, v in ins.items() if k in columns}
            res.append(ins)
        return res

    def read_and_preprocess(self, tokenizer, seed=None):
        raw_data = json.load(
            open(os.path.join(self.args.data_path, self.args.dataset_name, 'data_with_marker_tacred.json')))

        train_data = {}
        val_data = {}
        test_data = {}

        if seed is not None:
            random.seed(seed)

        cnt = 0
        for label in tqdm(raw_data.keys(), desc="Load TACRED data:"):
            cur_data = raw_data[label]
            random.shuffle(cur_data)
            train_raw_data = {"sentence": [], "labels": []}
            test_raw_data = {"sentence": [], "labels": []}
            train_count, test_count = 0, 0
            for idx, sample in enumerate(cur_data):
                sample["tokens"] = ' '.join(sample["tokens"])
                sample["relation"] = sample["relation"]
                if idx < len(cur_data) // 5 and test_count <= 40:
                    test_count += 1
                    test_raw_data["sentence"].append(sample["tokens"])
                    test_raw_data["labels"].append(sample["relation"])
                else:
                    train_count += 1
                    train_raw_data["sentence"].append(sample["tokens"])
                    train_raw_data["labels"].append(sample["relation"])
                    if train_count >= 320:
                        break
            cnt += test_count

            train_data[label] = self.preprocess(train_raw_data, tokenizer)
            test_data[label] = self.preprocess(test_raw_data, tokenizer)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
