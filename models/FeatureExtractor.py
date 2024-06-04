import copy
import logging

import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import BertModel

logger = logging.getLogger(__name__)


class PeftFeatureExtractor(nn.Module):
    """
    Extracting feature from the pretrained language model with little trainable parameters (peft)
    based on the different extract modes.
    """

    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.dataset = config.dataset_name

        self.bert = BertModel.from_pretrained(config.model_name_or_path)
        self.bert.resize_token_embeddings(
            self.bert.config.vocab_size + config.additional_special_tokens_len
        )

        self.origin_bert = None
        self.peft_bert = None
        self.peft_type = config.peft_type if hasattr(config, "peft_type") else None
        self.peft_init = config.peft_init if hasattr(config, "peft_init") else None

        self.prompts = nn.ParameterList()
        self.pre_seq_len = config.pre_seq_len if hasattr(config, "pre_seq_len") else None
        self.n_layer = self.bert.config.num_hidden_layers
        self.n_head = self.bert.config.num_attention_heads
        self.n_embd = self.bert.config.hidden_size // self.bert.config.num_attention_heads
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        if config.task_name == "RelationExtraction":
            self.extract_mode = "entity_marker"
        else:
            raise NotImplementedError

        if config.frozen:
            logger.info("freeze the parameters of the pretrained language model.")
            for param in self.bert.parameters():
                param.requires_grad = False
            self.origin_bert = copy.deepcopy(self.bert)
            for param in self.origin_bert.parameters():
                param.requires_grad = False

    def add_adapter(self, task_id):
        # Todo: support more peft types like prefix tuning, prompt tuning and so on.
        if self.peft_type == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1,
                target_modules=["key", "query", "value"],
            )
            adapter_name = f"task-{task_id}"
            self.peft_bert = get_peft_model(copy.deepcopy(self.bert), peft_config, adapter_name)
            self.peft_bert.print_trainable_parameters()
            logger.info(f"inject {self.peft_type} into the pretrain model, name is {adapter_name}")
        elif self.peft_type == "prefix":
            for param in self.prompts.parameters():
                param.requires_grad = False
            if len(self.prompts) > 0 and self.peft_init == "last":
                new_prompt = self.prompts[-1].data.clone()
            else:
                new_prompt = torch.randn(
                    self.pre_seq_len, self.n_layer * self.hidden_size * 2, device=self.device
                )
            self.prompts.append(nn.Parameter(new_prompt, requires_grad=True))
            logger.info(f"inject {self.peft_type} into the pretrain model")
        elif self.peft_type == "prompt":
            for param in self.prompts.parameters():
                param.requires_grad = False
            new_prompt = torch.randn(self.pre_seq_len, self.hidden_size, device=self.device)
            self.prompts.append(nn.Parameter(new_prompt, requires_grad=True))
            logger.info(f"inject {self.peft_type} into the pretrain model")
        else:
            raise NotImplementedError

    def save_and_load_all_adapters(self, task_id, save_dir, save=True):
        if self.peft_type == "lora":
            if save:
                self.peft_bert.save_pretrained(save_dir)
            self.peft_bert = PeftModel.from_pretrained(
                copy.deepcopy(self.bert),
                f"{save_dir}/task-0",
                adapter_name="task-0"
            )
            for i in range(1, task_id + 1):
                self.peft_bert.load_adapter(f"{save_dir}/task-{i}", adapter_name=f"task-{i}")

    def load_adapter(self, task_id):
        if self.peft_type == "lora":
            adapter_name = f"task-{task_id}"
            self.peft_bert.set_adapter(adapter_name)
        else:
            raise NotImplementedError

    def get_prompts_by_indices(self, indices, attention_mask):
        batch_size, _ = attention_mask.size()

        prompt_attention_mask = torch.ones(batch_size, self.pre_seq_len, dtype=torch.long, device=self.device)
        attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)

        prompt = torch.stack([self.prompts[idx] for idx in indices])
        past_key_values = None
        if self.peft_type == "prefix":
            past_key_values = prompt.view(batch_size, self.pre_seq_len, self.n_layer * 2, self.n_head, self.n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values, attention_mask, prompt

    def forward(
            self,
            input_ids,
            inputs_embeds=None,
            attention_mask=None,
            extract_mode=None,
            use_origin=False,
            indices=None,
            **kwargs
    ):
        batch_size, _ = input_ids.size()

        if attention_mask is None:
            attention_mask = input_ids != 0

        if use_origin:
            outputs = self.origin_bert(
                input_ids,
                attention_mask=attention_mask,
            )
        elif self.peft_type is not None and indices is not None:
            if self.peft_type == "lora":
                self.load_adapter(indices[0])
                outputs = self.peft_bert(
                    input_ids,
                    attention_mask=attention_mask,
                )
            elif self.peft_type == "prefix":
                past_key_values, attention_mask, _ = self.get_prompts_by_indices(indices, attention_mask)
                outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values
                )
            elif self.peft_type == "prompt":
                _, attention_mask, prompt = self.get_prompts_by_indices(indices, attention_mask)
                prompt_len = prompt.shape[1]
                inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
                inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)  # (batch, prompt_len + sent_len, dim)
                outputs = self.bert(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                outputs[0] = outputs[0][:, prompt_len:, :]
                attention_mask = attention_mask[:, prompt_len:]
            else:
                raise NotImplementedError
        else:
            # only for tuning
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=kwargs["past_key_values"] if "past_key_values" in kwargs else None,
            )

        extract_mode = extract_mode if extract_mode is not None else self.extract_mode
        # different feature extraction modes
        if extract_mode == "cls":
            hidden_states = outputs[1]  # (batch, dim)
        elif extract_mode == "mean_pooling":
            # (batch, dim)
            hidden_states = torch.sum(outputs[0] * attention_mask.unsqueeze(-1), dim=1) / \
                            torch.sum(attention_mask, dim=1).unsqueeze(-1)
        elif extract_mode == "mask":
            mask_pos = kwargs["mask_pos"]
            last_hidden_states = outputs[0]
            idx = torch.arange(last_hidden_states.size(0)).to(last_hidden_states.device)
            hidden_states = last_hidden_states[idx, mask_pos]
        elif extract_mode == "entity":
            last_hidden_states = outputs[0]
            subj_st, subj_ed = kwargs["subject_st"], kwargs["subject_ed"]
            obj_st, obj_ed = kwargs["object_st"], kwargs["object_ed"]
            hidden_states = []
            for idx in range(last_hidden_states.size(0)):
                subj = last_hidden_states[idx][subj_st[idx]: subj_ed[idx] + 1]
                obj = last_hidden_states[idx][obj_st[idx]: obj_ed[idx] + 1]
                subj = subj.mean(0)
                obj = obj.mean(0)
                hidden_states.append(torch.cat([subj, obj]))
            hidden_states = torch.stack(hidden_states, dim=0)
        elif extract_mode == "entity_marker":
            subject_start_pos = kwargs["subject_marker_st"]
            object_start_pos = kwargs["object_marker_st"]
            last_hidden_states = outputs[0]
            idx = torch.arange(last_hidden_states.size(0)).to(last_hidden_states.device)
            ss_emb = last_hidden_states[idx, subject_start_pos]
            os_emb = last_hidden_states[idx, object_start_pos]
            hidden_states = torch.cat([ss_emb, os_emb], dim=-1)  # (batch, 2 * dim)
        else:
            raise NotImplementedError

        return hidden_states
