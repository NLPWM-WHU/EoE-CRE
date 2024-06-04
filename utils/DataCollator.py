from typing import List, Dict, Any, Optional, Union

import torch
from attr import dataclass
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


@dataclass
class CustomCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def pad_to_same_length(self, batch_data):
        if isinstance(batch_data[0], int):
            if self.return_tensors == "pt":
                return torch.LongTensor(batch_data)
            else:
                return batch_data
        max_length = max([len(c) for c in batch_data])
        ans = []
        for ins in batch_data:
            ins = ins + [0] * (max_length - len(ins))
            ans.append(ins)
        if self.return_tensors == "pt":
            return torch.LongTensor(ans)
        else:
            return ans

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_keys = features[0].keys()
        batch = {k: [] for k in batch_keys}
        for ins in features:
            for k in batch_keys:
                batch[k].append(ins[k])
        for k in batch_keys:
            batch[k] = self.pad_to_same_length(batch[k])
        return batch
