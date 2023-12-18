#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import random
import sys
import warnings

import datasets
import torch
import transformers
from transformers import set_seed
from datasets import load_dataset

from accelerate import Accelerator
from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    apply_chat_template,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from trl import SFTTrainer


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")
warnings.filterwarnings("ignore", category=UserWarning, module="trl.trainer.ppo_config")


###########################################################################################
# Added this functionality for converting the dataset to the required format as
# specified by scripts/README.md. 
def format_input(inp, gpt_sw3=False):
    inp = inp.lstrip("\n")
    for role in ["system", "user", "assistant"]:
        if inp.startswith(f"<|im_start|>{role}"):
            inp = inp.replace(f"<|im_start|>{role}", "")
            if gpt_sw3:
                if role in ["assistant", "user"]:
                    role = "bot" if role == "assistant" else role
                    role = role.capitalize() + ":"
            return {"role": role, "content": inp.rstrip("\n").lstrip("\n")}


def format_inputs(row, gpt_sw3=False):
    """Convert a dataset row to the required format as described in the README.md"""
    inputs = row["input"].split("<|im_end|>")
    inputs[-1] += row["output"]
    return {"messages": [format_input(inp, gpt_sw3=gpt_sw3) for inp in inputs]}
###########################################################################################


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()
    gpt_sw3 = "gpt-sw3" in model_args.model_name_or_path

    # Set seed for reproducibility
    set_seed(training_args.seed)

    accelerator = Accelerator()

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ###############
    # Load datasets
    ###############
    if not isinstance(data_args.dataset_mixer, str):
        raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    
    ############################## for custom dataset #####################################
    else:
        base_path = data_args.dataset_mixer
        raw_datasets = load_dataset("json", data_files={"train": base_path + "train.json", "test": base_path + "eval.json"})
        raw_datasets = raw_datasets.map(format_inputs, fn_kwargs={"gpt_sw3": gpt_sw3})
        raw_datasets = raw_datasets.remove_columns(column_names=["input", "output"])

        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].shuffle(seed=training_args.seed)
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

        if data_args.max_eval_samples is not None:
            raw_datasets["test"] = raw_datasets["test"].shuffle(seed=training_args.seed)
            raw_datasets["test"] = raw_datasets["test"].select(range(data_args.max_eval_samples))
    #######################################################################################
    
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)
    tokenizer.padding_side = data_args.truncation_side
    if gpt_sw3:
        tokenizer.chat_template = """<|endoftext|><s>\n{% for message in messages %}{{message['role'] + '\n' + message['content'] + '\n<s>\n'}}{% endfor %}"""

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "sft"})
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map(),
        quantization_config=get_quantization_config(model_args),
    )
    logger.info("*** Model loaded! ***")

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        peft_config=get_peft_config(model_args),
        neftune_noise_alpha=training_args.neftune_noise_alpha,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {"finetuned_from": model_args.model_name_or_path}
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

        if training_args.push_to_hub is True:
            logger.info("Pushing to hub...")
            trainer.push_to_hub()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
