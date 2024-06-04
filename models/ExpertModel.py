from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from attr import dataclass

from models import PeftFeatureExtractor
from transformers.modeling_outputs import ModelOutput


@dataclass
class ExpertOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class ExpertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device

        self.feature_extractor = PeftFeatureExtractor(config)
        self.hidden_size = self.feature_extractor.bert.config.hidden_size

        self.num_old_labels = 0
        self.num_labels = 0

        self.classifier_hidden_size = self.feature_extractor.bert.config.hidden_size
        if config.task_name == "RelationExtraction":
            self.classifier_hidden_size = 2 * self.feature_extractor.bert.config.hidden_size

        self.classifier = nn.Linear(self.classifier_hidden_size, self.num_labels)

    @torch.no_grad()
    def new_task(self, num_labels):
        self.num_old_labels = self.num_labels
        self.num_labels += num_labels
        w = self.classifier.weight.data.clone()
        b = self.classifier.bias.data.clone()
        self.classifier = nn.Linear(self.classifier_hidden_size, self.num_labels, device=self.device)
        self.classifier.weight.data[:self.num_old_labels] = w
        self.classifier.bias.data[:self.num_old_labels] = b

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        hidden_states = self.feature_extractor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ExpertOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )
