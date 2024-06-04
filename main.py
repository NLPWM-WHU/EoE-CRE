import logging
import os
import random
import sys
from types import SimpleNamespace

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, set_seed

from data import FewRelData, TACREDData
from models import ExpertModel, EoE
from trainers import BaseTrainer, ExpertTrainer, EoETrainer

logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = "false"

task_to_data_reader = {
    "FewRel": FewRelData,
    "TACRED": TACREDData,
}

task_to_model = {
    "ExpertModel": ExpertModel,
    "EoE": EoE,
}

task_to_additional_special_tokens = {
    "RelationExtraction": ["[E11]", "[E12]", "[E21]", "[E22]"]
}

task_to_trainer = {
    "BaseTrainer": BaseTrainer,
    "ExpertTrainer": ExpertTrainer,
    "EoETrainer": EoETrainer,
}


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    args = OmegaConf.create()  # cfg seems to be read-only
    args = OmegaConf.merge(args, cfg.task_args, cfg.training_args)
    args = SimpleNamespace(**args)
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    logging.basicConfig(
        format="%(asctime)s - %(le5velname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(logging.INFO)

    additional_special_tokens = task_to_additional_special_tokens[args.task_name] \
        if args.task_name in task_to_additional_special_tokens else []
    args.additional_special_tokens = additional_special_tokens
    args.additional_special_tokens_len = len(additional_special_tokens)

    logger.info(f"additional special tokens: {additional_special_tokens}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
        additional_special_tokens=additional_special_tokens,
    )

    exp_results = []
    # conduct num_exp_rounds experiments and then calculate the average results
    for exp_idx in range(args.num_exp_rounds):
        exp_seed = args.seed + exp_idx * 100
        set_seed(exp_seed)

        data = task_to_data_reader[args.dataset_name](args)

        label_list = data.label_list
        task_seq = list(range(len(label_list)))
        if len(task_seq) != args.num_tasks * args.class_per_task:
            task_seq.extend([-1] * (args.num_tasks * args.class_per_task - len(task_seq)))
            random.shuffle(task_seq)
            task_seq = np.array(task_seq)
        else:
            random.shuffle(task_seq)
            task_seq = np.argsort(task_seq)
        if isinstance(args.class_per_task, int):
            task_seq = task_seq.reshape((args.num_tasks, args.class_per_task)).tolist()
        elif isinstance(args.class_per_task, list):
            tmp_seq = []
            cur = 0
            for n in args.class_per_task:
                tmp_seq.append(task_seq[cur:cur + n].tolist())
                cur += n
            task_seq = tmp_seq

        data.read_and_preprocess(tokenizer, seed=exp_seed)

        model = task_to_model[args.model_name](args)
        model.to(args.device)

        trainer = task_to_trainer[args.trainer_name](args=args)

        exp_result = trainer.run(
            data=data,
            model=model,
            tokenizer=tokenizer,
            label_order=task_seq,
            seed=exp_seed
        )
        exp_results.append(exp_result)
    # calculate the average results
    for k in exp_results[0].keys():
        avg_exp_results = [0] * args.num_tasks
        std_exp_results = [0] * args.num_tasks
        for idx in range(args.num_tasks):
            c = [e[k][idx] * 100 for e in exp_results]
            avg_exp_results[idx] = sum(c) / len(exp_results)
            avg_exp_results[idx] = round(avg_exp_results[idx], 2)
            std_exp_results[idx] = float(np.std(c))
        logger.info(f"{k} average : {avg_exp_results}")
        logger.info(f"{k}  std    : {std_exp_results}")
    logger.info("Training end !")


if __name__ == "__main__":
    main()
