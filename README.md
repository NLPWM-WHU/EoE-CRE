# An Ensemble-of-Experts Framework for Rehearsal-free Continual Relation Extraction

Code and data of our paper "An Ensemble-of-Experts Framework for Rehearsal-free Continual Relation Extraction" accepted by Findings of ACL 2024.

## Usage of Code
### 1.1 Environment Reuqirement

Python=3.9.18

```bash
pip install -r requirements.txt
```

### 1.2 train the first task
```bash
python main.py \
  +task_args=<DATASET> \
  +training_args=Expert \
  task_args.model_name_or_path=<MODEL_PATH> \
  task_args.config_name=<MODEL_PATH> \
  task_args.tokenizer_name=<MODEL_PATH>
```

### 1.3 train subsequent tasks
```bash
python main.py \
  +task_args=<DATASET> \
  +training_args=EoE \
  task_args.model_name_or_path=<MODEL_PATH> \
  task_args.config_name=<MODEL_PATH> \
  task_args.tokenizer_name=<MODEL_PATH>
```

`Note that <DATASET> denotest the datasets [FewRel, TACRED], <MODEL_PATH> denotes the path of "bert-base-uncased".
`