import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import HfArgumentParser
from trl import GRPOConfig

from trainer.grpo_trainer import GRPOTrainer
from utils.rewards import compute_score
from dataset.dataset import kk_Dataset
import os

# os.environ["VLLM_USE_V1"] = "0"
# /mnt/ssd3/chunhui/openr1/lib/python3.11/site-packages/transformers/utils/hub.py:106: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    they are equal to the arguments like parser.add_argument('--config_path', type=str) in argparse
    """

    config_path: Optional[str] = field(default=None, metadata={"help": "config path"})
    model_name_or_path : Optional[str] = field(default=None, metadata={"help": "model name or path"})
    out_dir: Optional[str] = field(default=None, metadata={"help": "output dir for model"})
    data_file: Optional[str] = field(default=None, metadata={"help": "train data file"})
    use_wandb: Optional[str] = field(default="flase", metadata={"help": "whether use wandb to report logs"})

    def __post_init__(self):
        if self.config_path is None:
            raise ValueError("config path should not none")


def main():
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0] # index 0 because it returns a tuple of length 1; potentially could be more than one dataclass and then unpacking would be useful
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s") # to show the logs of this file
    transformers.logging.set_verbosity_info() # to show the logs of transformers
    logging.info(data_args) # to show the arguments passed to this file

    # reward_funcs_registry = {"accuracy": accuracy_reward, "format": format_reward}
    # reward_funcs = [reward_funcs_registry["accuracy"], reward_funcs_registry["format"]]
    reward_funcs_registry = {"compute_score": compute_score}
    reward_funcs = [reward_funcs_registry["compute_score"]]

    train_dataset = kk_Dataset(data_args.data_file)
    training_args = GRPOConfig(
    seed=42,
    data_seed=42,
    output_dir=data_args.out_dir,
    deepspeed=data_args.config_path,
    max_prompt_length=64,
    max_completion_length=128,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=5,
    logging_steps=1,
    bf16=True,
    report_to="wandb" if data_args.use_wandb == "true" else [],
    gradient_checkpointing=False,
    num_train_epochs=1,
    max_steps=1000,
    run_name="KK-GRPO",
    save_steps=100,
    save_only_model=True,   
    temperature=1,
    num_generations=2,  # 先调小，保证跑起来
    vllm_device="auto",
    loss_type="dr_grpo"
    )
    
    # display the training arguments user-friendly
    logging.info("Training arguments:")
    for arg, value in vars(training_args).items():
        logging.info(f"{arg}: {value}")
    trainer = GRPOTrainer(
        model=data_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None) # TODO: add eval dataset; 

    trainer.train()
    trainer.save_model(data_args.out_dir)


if __name__ == "__main__":
    main()
