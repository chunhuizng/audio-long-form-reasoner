# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import contextlib
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence # Assume pad is pad_sequence
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from accelerate.state import DistributedType

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, selective_log_softmax
from trl.trainer.callbacks import SyncRefModelCallback

# vLLM support
from unittest.mock import patch
from vllm import LLM, SamplingParams
import deepspeed
import json
from trainer.pad_tools import pad
from accelerate.utils.other import is_compiled_module


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from trainer.modelling_audio_only_thinker import AudioOnlyThinker
from trainer.audio_only_processor import AudioOnlyProcessor
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


# Based on R1-V code base, https://github.com/Deep-Agent/R1-V/blob/main/src/r1-v/src/open_r1/trainer/grpo_trainer.py
class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "sdpa"
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        vllm_model = None
        default_limits = None
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation # We hope audio qwen2 model can use this implementation...
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # display model_init_kwargs user-friendly
            if isinstance(model_init_kwargs, dict):
                model_init_kwargs_str = json.dumps(model_init_kwargs, indent=4)
                model_init_kwargs_str = textwrap.indent(model_init_kwargs_str, " " * 4)
                print(f"model_init_kwargs: {model_init_kwargs_str}")
                
            if "AudioOnlyThinker" in model_id or "audio-thinker" in model_id:
                model = AudioOnlyThinker.from_pretrained(model, torch_dtype=torch.bfloat16, **model_init_kwargs)
                vllm_model = "Qwen/Qwen2.5-Omni-7B"
                default_limits = {"audio": 1}
            elif "Qwen2-Audio" in model_id:
                model = Qwen2AudioForConditionalGeneration.from_pretrained(model, torch_dtype=torch.bfloat16, **model_init_kwargs)
                vllm_model = "Qwen/Qwen2-Audio-7B"
                default_limits = {"image": 0, "video": 0, "audio": 1}
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "AudioOnlyThinker" in model_id or "audio-thinker" in model_id:
                self.ref_model = AudioOnlyThinker.from_pretrained(model_id, torch_dtype=torch.bfloat16, **model_init_kwargs)
            elif "Qwen2-Audio" in model_id:
                print("model_init_kwargs:", model_init_kwargs)
                self.ref_model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif not is_peft_model(model):
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "AudioOnlyThinker" in model_id or "audio-thinker" in model_id:
                processing_class = AudioOnlyProcessor.from_pretrained(model_id)#, padding_side="left")
                processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                processing_class.tokenizer.padding_side  = 'left'
                # processing_class.feature_extractor.padding_side  = 'left'
                print(f"Processing class: {processing_class}")
            elif "Qwen2-Audio" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                processing_class.tokenizer.padding_side  = 'left'
                print(f"Processing class: {processing_class}")
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path, padding_side="left")
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        set_seed(args.seed, device_specific=True)

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=processing_class.pad_token_id,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        if self.accelerator.is_main_process:
            # load vllm
            vllm_device = args.vllm_device
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible_devices:
                num_gpus = len(visible_devices.split(","))
            else:
                num_gpus = torch.cuda.device_count()
            if vllm_device == "auto":
                vllm_device = f"cuda:{num_gpus - 1}"  # safest way to always use last visible GPU
            # Check that the requested device is available
            if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                raise ValueError(
                    f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                    "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                    "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                    f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                )
            # Check that the requested device is not also used for training
            if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                print(
                    f"The requested device {vllm_device} is also used for training. This may lead to unexpected "
                    "behavior. It is recommended to use a dedicated device for vLLM."
                )
            # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
            # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
            # setting (profiling_patch).
            world_size_patch = patch("torch.distributed.get_world_size", return_value=1)

            profiling_patch = patch(
                "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
            )
            with world_size_patch, profiling_patch:
                self.llm = LLM(
                    model=vllm_model,
                    device=vllm_device,
                    gpu_memory_utilization=0.6,
                    enable_prefix_caching=True,
                    limit_mm_per_prompt={"image": 0, "video": 0, "audio": 1},
                    # mm_processor_kwargs={
                    #     "sampling_rate": 16000,
                    #     # "normalize": True  
                    # }
                )
                self.sampling_params = SamplingParams(
                    temperature=0.9,
                    top_p=0.9,
                    top_k=50,
                    max_tokens=self.max_completion_length,
                )

        self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        self.accelerator.wait_for_everyone()
                
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, features_values, features_masks):
        # print(f"input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, features_values: {features_values.shape}, features_masks: {features_masks.shape}")
        logits = model(input_ids, attention_mask=attention_mask, input_features=features_values, feature_attention_mask=features_masks).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # vllm
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device

        # 1. Process inputs consistently (get original IDs, masks, features)
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        audios = [x["audio"] for x in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            audios=audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            padding_side="left", # Important for concatenation later
            add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        # Store original (untruncated) prompt components before repetition
        original_prompt_ids = prompt_inputs["input_ids"]
        original_prompt_attention_mask = prompt_inputs["attention_mask"]
        original_input_features = prompt_inputs["input_features"]
        original_feature_attention_mask = prompt_inputs["feature_attention_mask"]

        # --- vLLM Generation Block ---
        inputs_vllm = []
        # build inputs_vllm here
        for example in inputs:
            prompt = self.processing_class.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=True
            )
            waveform = np.asarray(example["audio"], dtype=np.float32)
            multi_modal_data = [(waveform, example["sample_rate"])]
            for _ in range(self.num_generations):
                inputs_vllm.append({
                    "prompt": prompt,
                    "multi_modal_data": {"audio": multi_modal_data},
                })

        # vLLM model weight sync/load logic here
        if self.state.global_step != self._last_loaded_step:
            print(f"[Rank {self.accelerator.process_index}] Syncing weights from step {self.state.global_step} to vLLM.")
            # Gather parameters if using DeepSpeed ZeRO stage 3 or FSDP
            needs_gathering = (
                self.accelerator.state.distributed_type == DistributedType.DEEPSPEED and self.accelerator.state.deepspeed_plugin.zero_stage == 3
            ) or self.accelerator.state.distributed_type == DistributedType.FSDP

            # Use GatheredParameters only when necessary and if DeepSpeed is used
            context_manager = deepspeed.zero.GatheredParameters(model.parameters(), enabled=needs_gathering) if self.accelerator.state.distributed_type == DistributedType.DEEPSPEED else torch.no_grad()

            with context_manager:
                # Unwrap the model to get the base Hugging Face model
                # Handles DDP, FSDP, DeepSpeed, torch.compile
                unwrapped_model = self.accelerator.unwrap_model(model)
                # Get the state dict from the unwrapped model
                state_dict = unwrapped_model.state_dict()

                # --- Map state_dict keys ---
                # Map keys from AudioOnlyThinker structure (HF Trainer) to vLLM's Qwen2_5OmniThinker structure
                mapped_state_dict = {}
                skipped_keys = []
                print(f"[Rank {self.accelerator.process_index}] Starting state_dict mapping...") # Debug print
                for key, value in state_dict.items():
                    new_key = key
                    if key.startswith("model."):
                        # Map Trainer's 'model' (Qwen2_5OmniThinkerTextModel) -> vLLM's 'language_model.model' (Qwen2Model)
                        new_key = "language_model." + key # Prepend 'language_model.'
                    elif key.startswith("lm_head."):
                        # Map Trainer's 'lm_head' (Linear) -> vLLM's 'language_model.lm_head' (ParallelLMHead)
                        new_key = key.replace("lm_head.", "language_model.lm_head.", 1)
                    elif key.startswith("audio_tower."):
                        # Assume audio_tower structure is compatible between trainer model and vLLM model
                        new_key = key # Keep as is
                    else:
                        # Log unexpected keys that don't belong to audio_tower, model, or lm_head
                        skipped_keys.append(key)
                        continue # Skip keys we don't know how to map

                    # Store the mapped key and original value (tensor)
                    # mapped_state_dict[new_key] = value.cpu() # Move tensor to CPU before sending? vLLM load_weights might handle device placement. Check vLLM docs. Let's keep it on original device for now.
                    target_device = self.accelerator.device  # or "cuda" / "cpu" depending on vLLM setup
                    mapped_state_dict[new_key] = value.to(target_device)
                    # mapped_state_dict[new_key] = value

                if skipped_keys:
                    print(f"[Rank {self.accelerator.process_index}] Skipped mapping for {len(skipped_keys)} keys: {skipped_keys[:5]}...") # Print only a few skipped keys
                print(f"[Rank {self.accelerator.process_index}] Finished state_dict mapping. Mapped {len(mapped_state_dict)} keys.") # Debug print

                # --- Load weights into vLLM ---
                # vLLM typically runs its engine on rank 0 or a dedicated process.
                # Weight loading should likely only happen on the process controlling vLLM.
                if self.accelerator.is_main_process:
                    print(f"[Rank {self.accelerator.process_index}] Is main process. Attempting to load mapped weights into vLLM.")
                    try:
                        # Access the vLLM model instance
                        # This path might need adjustment based on your specific vLLM integration
                        # Ensure self.llm and the subsequent attributes exist and point to the vLLM engine/model
                        if not hasattr(self, "llm") or not hasattr(self.llm, "llm_engine"):
                            raise AttributeError("Cannot find vLLM engine via self.llm.llm_engine")

                        # vLLM engine might have different ways to access the underlying model(s) depending on setup
                        # Common pattern: engine -> model_executor -> driver_worker (if distributed) -> model_runner -> model
                        model_executor = getattr(self.llm.llm_engine, "model_executor", None)
                        if not model_executor:
                            raise AttributeError("Cannot find model_executor in vLLM engine")

                        # Assuming access via driver_worker for potentially distributed vLLM setups
                        driver_worker = getattr(model_executor, "driver_worker", None)
                        if driver_worker and hasattr(driver_worker, "model_runner") and hasattr(driver_worker.model_runner, "model"):
                            llm_model = driver_worker.model_runner.model
                            print(f"[Rank {self.accelerator.process_index}] Found vLLM model via driver_worker.")
                        else:
                            # Fallback: Maybe model_executor holds the model directly in simpler setups?
                            if hasattr(model_executor, "model"): # Needs verification based on vLLM version/setup
                                llm_model = model_executor.model
                                print(f"[Rank {self.accelerator.process_index}] Found vLLM model via model_executor.")
                            else:
                                raise AttributeError("Could not reliably access the vLLM model runner's model instance.")


                        # Use vLLM's load_weights method, passing the mapped state dictionary items
                        # This method should handle internal conversions (e.g., HF Linear -> vLLM ParallelLinear)
                        print(f"[Rank {self.accelerator.process_index}] Calling vLLM load_weights...")
                        llm_model.load_weights(mapped_state_dict.items())
                        print(f"[Rank {self.accelerator.process_index}] Successfully called vLLM load_weights.")

                    except AttributeError as e:
                        print(f"[Rank {self.accelerator.process_index}] ERROR: Failed to access vLLM model components: {e}. Weight sync aborted.")
                    except Exception as e:
                        # Catch potential errors during the load_weights call itself (e.g., key mismatches vLLM can't handle)
                        print(f"[Rank {self.accelerator.process_index}] ERROR: Exception during vLLM load_weights: {type(e).__name__} - {e}. Weight sync aborted.")
                        # You might want more detailed debugging here if load_weights fails internally
                        # E.g., compare keys again: print(list(mapped_state_dict.keys())[:10], list(llm_model.state_dict().keys())[:10])

            # Update the step counter *after* the synchronization attempt
            self._last_loaded_step = self.state.global_step

            # Optional: Barrier to ensure all processes wait for rank 0 to finish loading before proceeding
            # This might be important if subsequent operations depend on vLLM having the updated weights
            print(f"[Rank {self.accelerator.process_index}] Waiting for other processes...")
            self.accelerator.wait_for_everyone()
            print(f"[Rank {self.accelerator.process_index}] Finished weight sync block.")

        # Generate completions with vLLM
        all_inputs_vllm = gather_object(inputs_vllm)
        if self.accelerator.is_main_process:
            outputs = self.llm.generate(all_inputs_vllm, sampling_params=self.sampling_params, use_tqdm=False)
            completion_ids_list = [out.token_ids for completions in outputs for out in completions.outputs]
        else:
            completion_ids_list = [None] * len(all_inputs_vllm)

        completion_ids_list = broadcast_object_list(completion_ids_list, from_process=0)
        process_slice = slice(
            self.accelerator.process_index * len(prompts) * self.num_generations,
            (self.accelerator.process_index + 1) * len(prompts) * self.num_generations,
        )
        completion_ids_list = completion_ids_list[process_slice]
        completion_ids_list = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_ids_padded = pad(completion_ids_list, padding_value=self.processing_class.pad_token_id)
        # --- End vLLM Generation Block ---
        # 2. Prepare inputs for the forward pass (policy model logps)
        #    Repeat the *original, untruncated* prompt info and features
        prompt_ids_repeated = original_prompt_ids.repeat(self.num_generations, 1)
        prompt_mask_repeated = original_prompt_attention_mask.repeat(self.num_generations, 1)
        features_repeated = original_input_features.repeat(self.num_generations, 1, 1)
        feature_mask_repeated = original_feature_attention_mask.repeat(self.num_generations, 1) # Check if dim=0 or dim=1 is correct for this mask based on its meaning

        # 3. Construct the full sequence ID and Mask for the forward pass
        #    **Crucially, use the untruncated, repeated prompt IDs**
        final_input_ids = torch.cat([prompt_ids_repeated, completion_ids_padded], dim=1)

        # Create completion mask based on EOS in completion_ids_padded
        is_eos = completion_ids_padded == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Construct final attention mask using untruncated prompt mask + completion mask
        final_attention_mask = torch.cat([prompt_mask_repeated, completion_mask], dim=1)

        # --- Micro-batching Setup ---
        # Determine the full batch size after repeating for generations
        full_batch_size = final_input_ids.shape[0] # Should be B * self.num_generations
        # Define micro-batch size (tune this based on memory)
        # Example: Process 2 samples at a time if memory allows
        # Or set based on original batch size: micro_batch_size = len(inputs) # Process original batch size `B` at a time
        micro_batch_size = getattr(self, "per_device_train_batch_size", 1) # A reasonable default, maybe too small. Or set explicitly: micro_batch_size = 2 # Try 2
        if full_batch_size == 0:
            # Handle empty batch case if necessary
            return torch.tensor(0.0, device=device, requires_grad=True) # Or appropriate handling

        print(f"[Rank {self.accelerator.process_index}] Full batch size for logp calculation: {full_batch_size}, Micro-batch size: {micro_batch_size}")

        # --- Helper function for micro-batched logp calculation ---
        def get_logps_micro_batch(target_model, mb_size):
            target_model.eval() # Ensure model is in eval mode for logp calculation if it involves dropout etc.
            all_logps = []
            for start_idx in range(0, full_batch_size, mb_size):
                end_idx = min(start_idx + mb_size, full_batch_size)
                # Slice the inputs for the current micro-batch
                mb_input_ids = final_input_ids[start_idx:end_idx]
                mb_attention_mask = final_attention_mask[start_idx:end_idx]
                mb_features = features_repeated[start_idx:end_idx]
                mb_feature_mask = feature_mask_repeated[start_idx:end_idx]

                # Ensure tensors are on the correct device (should be handled by _prepare_inputs or model placement)
                mb_input_ids = mb_input_ids.to(device)
                mb_attention_mask = mb_attention_mask.to(device)
                mb_features = mb_features.to(device)
                mb_feature_mask = mb_feature_mask.to(device)

                print(f"[Rank {self.accelerator.process_index}] Calculating logps for micro-batch {start_idx//mb_size + 1}/{(full_batch_size + mb_size - 1)//mb_size} with shapes: IDs={mb_input_ids.shape}, Feat={mb_features.shape}")

                # Call the original logp function with the micro-batch
                # Use torch.no_grad() if gradients are not needed here (depends if policy model requires grads)
                # Assuming policy model needs grads, ref model doesn't
                if target_model is model: # Policy model
                    with torch.enable_grad(): # Ensure gradients are enabled if needed by policy loss
                        # Check if gradient checkpointing is enabled and beneficial
                        # model.gradient_checkpointing_enable() # Enable if not done elsewhere
                        logps = self._get_per_token_logps(target_model,
                                                        mb_input_ids,
                                                        mb_attention_mask,
                                                        mb_features,
                                                        mb_feature_mask)
                else: # Reference model
                    with torch.no_grad():
                        # ref_model.gradient_checkpointing_enable() # Enable if needed/possible
                        logps = self._get_per_token_logps(target_model,
                                                        mb_input_ids,
                                                        mb_attention_mask,
                                                        mb_features,
                                                        mb_feature_mask)

                all_logps.append(logps.cpu()) # Move to CPU to free GPU memory faster within the loop
                # Consider torch.cuda.empty_cache() here if fragmentation is severe, but use cautiously

            # Concatenate results from all micro-batches
            # Ensure concatenation happens on the correct device for subsequent calculations
            concatenated_logps = torch.cat(all_logps, dim=0).to(device)
            # Put model back to train mode if it was changed (Trainer usually handles this)
            # target_model.train()
            return concatenated_logps

        # 4. Get log probabilities for the policy model using micro-batching
        print(f"[Rank {self.accelerator.process_index}] Getting policy logps...")
        per_token_logps = get_logps_micro_batch(model, micro_batch_size)
        print(f"[Rank {self.accelerator.process_index}] Policy logps shape: {per_token_logps.shape}")

        prompt_length = prompt_ids_repeated.size(1) # Length of the untruncated prompt part
        # Adjust slicing based on _get_per_token_logps shift (which slices off first token logit and first input ID)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        # 5. Get reference model log probabilities using micro-batching
        print(f"[Rank {self.accelerator.process_index}] Getting reference logps...")
        ref_model_to_use = None
        disable_adapter_context = contextlib.nullcontext() # Use nullcontext if no adapter used

        if self.ref_model is not None:
            ref_model_to_use = self.ref_model
        else:
            # Need to handle the disable_adapter case carefully with micro-batching
            # Option 1: Disable adapter outside the loop (might not work if model is wrapped complexly)
            # Option 2: Pass model and apply context manager inside get_logps_micro_batch (cleaner)
            # Let's assume get_logps_micro_batch handles the no_grad context correctly for ref model
            ref_model_to_use = model
            # The disable_adapter context needs to wrap the forward pass inside _get_per_token_logps
            # Modify _get_per_token_logps or wrap the call within get_logps_micro_batch
            # For simplicity here, we assume _get_per_token_logps is called correctly for ref/base model
            # If using adapters, the context needs to be handled around the model call in _get_per_token_logps

            # We will modify get_logps_micro_batch slightly to handle this later if needed.
            # For now, assume get_logps_micro_batch correctly calls the reference.

        # --- Updated call to handle ref_model / adapter disabling ---
        def get_logps_micro_batch_ref(mb_size):
            all_logps = []
            for start_idx in range(0, full_batch_size, mb_size):
                end_idx = min(start_idx + mb_size, full_batch_size)
                mb_input_ids = final_input_ids[start_idx:end_idx]
                mb_attention_mask = final_attention_mask[start_idx:end_idx]
                mb_features = features_repeated[start_idx:end_idx]
                mb_feature_mask = feature_mask_repeated[start_idx:end_idx]

                print(f"[Rank {self.accelerator.process_index}] Calculating REF logps for micro-batch {start_idx//mb_size + 1}/{(full_batch_size + mb_size - 1)//mb_size}")

                with torch.inference_mode(): # Use inference_mode for ref model
                    if self.ref_model is not None:
                        ref_model_to_use = self.ref_model
                        logps = self._get_per_token_logps(ref_model_to_use, mb_input_ids, mb_attention_mask, mb_features, mb_feature_mask)
                    else:
                        # Use the base model (policy model) with adapters disabled
                        base_model = self.accelerator.unwrap_model(model) # Ensure we get the base model for adapter disabling
                        with base_model.disable_adapter(): # Apply context manager here
                            logps = self._get_per_token_logps(model, mb_input_ids, mb_attention_mask, mb_features, mb_feature_mask) # Call with original wrapped model is usually fine

                all_logps.append(logps.cpu())

            concatenated_logps = torch.cat(all_logps, dim=0).to(device)
            return concatenated_logps

        ref_per_token_logps = get_logps_micro_batch_ref(micro_batch_size)
        # -------------------------------------------------------------

        print(f"[Rank {self.accelerator.process_index}] Reference logps shape: {ref_per_token_logps.shape}")
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]


        # --- Rest of the loss calculation and metric logging ---
        # Compute the KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode completions for reward function (using completion_ids_padded)
        completions = self.processing_class.batch_decode(completion_ids_padded, skip_special_tokens=True)

        # Compute rewards
        prompts_for_reward = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts_for_reward), len(self.reward_funcs))
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ): # this step is to loop through the reward functions and reward processing classes
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]} # 构造 reward_kwargs，key 是除了 prompt 和 completion 之外的所有 key. 其余 dataset 字段 比如 "solution"、"audio_path"、"index" 会被收集进 reward_kwargs. 然后 每个字段被按 num_generations 重复，确保 reward_kwargs 里的每个字段长度与 prompts 和 completions 对齐。
            for key in reward_kwargs:
                for example in inputs:
                    # Repeat each value in the column for num_generations times; solution 实际是作为 reward_kwargs["solution"] 传进 reward_func 里的
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs) # 调用 reward function
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device) # 👉 这里把 reward_func 输出的分数 list 填充到 rewards_per_func 第 i 列，表示这是 第 i 个 reward function 的打分结果。
            # rewards_per_func v.s. reward_func: rewards_per_func is a tensor of shape (self.num_generations, number of reward functions), and reward_func is a function that takes prompts and completions as input and returns a list of rewards
                
        # Sum the rewards from all reward functions. (dim=1) means summing along the columns (i.e., the reward functions)
        rewards = rewards_per_func.sum(dim=1).to(device)

        # Compute advantages
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Compute loss
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # Apply completion mask to loss calculation (mask has shape [B*G, C])
        # per_token_loss has shape [B*G, C] where C is completion length
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        # Ensure the tensor is on the correct device before gathering
        rewards_per_func_on_device = rewards_per_func.to(self.accelerator.device)
        # Gather the tensor that is guaranteed to be on the correct device
        gathered_rewards = self.accelerator.gather_for_metrics(rewards_per_func_on_device)
        reward_per_func = gathered_rewards.mean(0)
        # reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss  
    
    def compute_loss_with_lip(self, model, inputs, return_outputs=False, num_items_in_batch=None, # Original arg, seems unused later
                     lambda_lip_consistency=0.1, # Weight for the new penalty
                     epsilon_lip_consistency=1e-6, # Epsilon for the new penalty denominator
                     ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device

        # 1. Process inputs consistently (get original IDs, masks, features)
        prompts = [x["prompt"] for x in inputs] # List of prompts, size B
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        audios = [x["audio"] for x in inputs] # Assuming audio modality
        prompt_inputs = self.processing_class(
            text=prompts_text,
            audios=audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            # padding_side="left", # Padding side might depend on model architecture needs
            add_special_tokens=False # Check if this is appropriate for the model
        )
        # Move processed inputs to device *before* _prepare_inputs potentially modifies them
        prompt_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in prompt_inputs.items()}
        # Assuming super()._prepare_inputs is needed for potential further device placement/type casting
        # If it's just an identity function as overridden later, this call might be skippable
        prompt_inputs = super()._prepare_inputs(prompt_inputs)


        original_prompt_ids = prompt_inputs["input_ids"] # Shape (B, P)
        original_prompt_attention_mask = prompt_inputs["attention_mask"] # Shape (B, P)
        original_input_features = prompt_inputs["input_features"] # Shape (B, F_len, F_dim)
        original_feature_attention_mask = prompt_inputs["feature_attention_mask"] # Shape (B, F_len)
        original_batch_size = original_prompt_ids.size(0) # B

        # --- vLLM Generation Block ---
        inputs_vllm = []
        for example in inputs:
            # Apply chat template if necessary for vLLM prompt format
            prompt_formatted = self.processing_class.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=True # Usually needed for generation
            )
            waveform = np.asarray(example["audio"], dtype=np.float32)
            # Assuming sample rate is needed by vLLM multi-modal input format
            multi_modal_data = [(waveform, 16000)] # Use the known sampling rate
            for _ in range(self.num_generations):
                inputs_vllm.append({
                    "prompt": prompt_formatted, # Use formatted prompt
                    "multi_modal_data": {"audio": multi_modal_data},
                })

        # vLLM model sync/load logic (seems specific to a particular setup)
        if self.state.global_step != self._last_loaded_step:
            # This block needs careful handling depending on the exact setup
            # with deepspeed.zero.GatheredParameters(model.parameters()): # Requires deepspeed
            # Ensure unwrapped_model is correctly obtained (may depend on DDP/FSDP/DeepSpeed wrapper)
            unwrapped_model = self.accelerator.unwrap_model(model)
            state_dict = ( # Handle compiled models if necessary
                unwrapped_model._orig_mod.state_dict()
                if hasattr(unwrapped_model, '_orig_mod') # Check if compiled
                else unwrapped_model.state_dict()
            )
            if self.accelerator.is_main_process:
                # This assumes direct access to vLLM engine internals
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict.items())
            self.accelerator.wait_for_everyone() # Ensure all processes wait for weights load
            self._last_loaded_step = self.state.global_step


        # Generate completions with vLLM
        all_inputs_vllm = gather_object(inputs_vllm) # Gather inputs from all processes to main
        if self.accelerator.is_main_process:
            outputs = self.llm.generate(all_inputs_vllm, sampling_params=self.sampling_params, use_tqdm=False)
            # Extract token IDs, outputs is List[RequestOutput]
            # each RequestOutput has List[CompletionOutput] in outputs
            completion_ids_list = [comp_out.token_ids for req_out in outputs for comp_out in req_out.outputs]
            # Add handling for potential errors or empty outputs if needed
        else:
            completion_ids_list = None # Placeholder on non-main processes

        # Broadcast results back to all processes
        completion_ids_list = broadcast_object_list(completion_ids_list, from_process=0)

        # Get the portion relevant to the current process
        # Each original input generated G completions. Total = B*G*world_size
        num_local_prompts = len(prompts) # B on this process
        total_generated_this_process = num_local_prompts * self.num_generations # B*G on this process
        # Ensure the list is not None before slicing/processing
        if completion_ids_list is None:
             completion_ids_list = [[]] * total_generated_this_process # Avoid errors, maybe raise instead?


        # Convert to tensors and pad
        completion_ids_tensors = [torch.tensor(ids, device=device, dtype=torch.long) for ids in completion_ids_list]
        # Pad completions generated by this process (B*G items)
        completion_ids_padded = pad_sequence(completion_ids_tensors, batch_first=True, padding_value=self.processing_class.pad_token_id)
        # completion_ids_padded shape: (B*G, C) where C is max completion length on this process
        # --- End vLLM Generation Block ---


        # 2. Prepare inputs for the forward pass (policy model logps)
        #    Repeat the *original, untruncated* prompt info and features G times
        prompt_ids_repeated = original_prompt_ids.repeat_interleave(self.num_generations, dim=0) # Shape (B*G, P)
        prompt_mask_repeated = original_prompt_attention_mask.repeat_interleave(self.num_generations, dim=0) # Shape (B*G, P)
        features_repeated = original_input_features.repeat_interleave(self.num_generations, dim=0) # Shape (B*G, F_len, F_dim)
        # Check dims for feature mask repetition - usually along batch dim 0
        feature_mask_repeated = original_feature_attention_mask.repeat_interleave(self.num_generations, dim=0) # Shape (B*G, F_len)

        # 3. Construct the full sequence ID and Mask for the forward pass
        final_input_ids = torch.cat([prompt_ids_repeated, completion_ids_padded], dim=1) # Shape (B*G, P+C)

        # Create completion mask based on EOS in completion_ids_padded
        is_eos = completion_ids_padded == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device) # Max length default
        eos_present = is_eos.any(dim=1)
        if eos_present.any(): # Avoid error on argmax for rows with no EOS
            eos_idx[eos_present] = is_eos.int().argmax(dim=1)[eos_present]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        # Mask includes the EOS token itself if present (use <= or < depending on desired behavior)
        # Using < excludes EOS from loss/logp sum if present
        completion_mask = (sequence_indices < eos_idx.unsqueeze(1)).int() # Shape (B*G, C)

        # Construct final attention mask
        final_attention_mask = torch.cat([prompt_mask_repeated, completion_mask], dim=1) # Shape (B*G, P+C)

        # 4. Get log probabilities using the consistent inputs
        policy_inputs_dict = {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_mask,
            "input_features": features_repeated, # Correct kwarg name? Check model's forward signature
            "feature_attention_mask": feature_mask_repeated
        }
        # Use the model passed as argument, which should be the trainable model
        per_token_logps = self._get_per_token_logps(model, **policy_inputs_dict)

        prompt_length = prompt_ids_repeated.size(1) # Length of the original prompt part (P)
        # Shift logps to align with completion tokens
        per_token_logps = per_token_logps[:, prompt_length - 1:] # Shape (B*G, C)

        # 5. Get reference model log probabilities
        with torch.inference_mode():
            ref_inputs_dict = policy_inputs_dict # Use same inputs structure
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, **ref_inputs_dict)
            else:
                # Assumes model has adapter disabling capability
                # Need to ensure model passed here is the one to disable adapters on
                with self.accelerator.unwrap_model(model).disable_adapter():
                     ref_per_token_logps = self._get_per_token_logps(model, **ref_inputs_dict)
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:] # Shape (B*G, C)


        # --- Rest of the loss calculation ---
        # Compute the KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1 # Shape (B*G, C)

        # Decode completions for reward function
        completions = self.processing_class.batch_decode(completion_ids_padded, skip_special_tokens=True)

        # Compute rewards
        prompts_for_reward = [p for p in prompts for _ in range(self.num_generations)] # Repeat original prompts B*G times
        rewards_per_func = torch.zeros(len(prompts_for_reward), len(self.reward_funcs), device=device) # Use device
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes) # Assume reward_processing_classes exists
        ):
            # Prepare kwargs for reward function, repeating B*G times
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion", "audio"]} # Adjust keys
            for key in reward_kwargs:
                for example in inputs: # Loop through original B inputs
                    reward_kwargs[key].extend([example[key]] * self.num_generations)

            # Call reward function - ensure prompts_for_reward and completions match length (B*G)
            output_reward_func = reward_func(prompts=prompts_for_reward, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum rewards and ensure on correct device
        rewards = rewards_per_func.sum(dim=1) # Shape (B*G)

        # Compute advantages
        rewards_grouped = rewards.view(original_batch_size, self.num_generations) # Shape (B, G)
        mean_grouped_rewards = rewards_grouped.mean(dim=1, keepdim=True) # Shape (B, 1)
        std_grouped_rewards = rewards_grouped.std(dim=1, keepdim=True) # Shape (B, 1)
        # Expand back to B*G
        mean_grouped_rewards_expanded = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards_expanded = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards_expanded) / (std_grouped_rewards_expanded + 1e-4) # Shape (B*G)

        # --- GRPO Loss Calculation ---
        # Policy gradient term (simplified PPO ratio style)
        policy_ratio = torch.exp(per_token_logps - per_token_logps.detach())
        policy_loss_per_token = policy_ratio * advantages.unsqueeze(1) # Shape (B*G, C)

        # Combine policy loss and KL divergence
        # Note: completion_mask may need .float() if used in division
        completion_mask_float = completion_mask.float() # Shape (B*G, C)
        per_token_loss_grpo = -(policy_loss_per_token - self.beta * per_token_kl) # Shape (B*G, C)

        # Calculate GRPO loss (masked mean over tokens, then mean over batch*generations)
        # Ensure division by sum of mask is safe (handle sequences with zero length completions if possible)
        sum_completion_mask = completion_mask_float.sum(dim=1)
        grpo_loss_unreduced = (per_token_loss_grpo * completion_mask_float).sum(dim=1) / torch.clamp(sum_completion_mask, min=1e-8) # Shape (B*G)
        grpo_loss = grpo_loss_unreduced.mean() # Scalar


        # --- [NEW CODE: Reward-Sensitive Log-Probability Smoothness Penalty] ---
        consistency_penalty = torch.tensor(0.0, device=device)
        if lambda_lip_consistency > 0 and self.num_generations > 1:
            # 1. Calculate sequence log-probabilities (using the same mask as the loss)
            sequence_logps = (per_token_logps * completion_mask_float).sum(dim=1) # Shape (B*G)

            # 2. Reshape rewards and sequence logps to (B, G) for grouped comparison
            # rewards_grouped is already calculated above: shape (B, G)
            sequence_logps_grouped = sequence_logps.view(original_batch_size, self.num_generations) # Shape (B, G)

            # 3. Calculate pairwise differences within each group (vectorized across batch B)
            rewards_expanded1 = rewards_grouped.unsqueeze(2) # Shape (B, G, 1)
            rewards_expanded0 = rewards_grouped.unsqueeze(1) # Shape (B, 1, G)
            reward_diffs_batch = rewards_expanded1 - rewards_expanded0 # Shape (B, G, G)

            logps_expanded1 = sequence_logps_grouped.unsqueeze(2) # Shape (B, G, 1)
            logps_expanded0 = sequence_logps_grouped.unsqueeze(1) # Shape (B, 1, G)
            logp_diffs_batch = logps_expanded1 - logps_expanded0 # Shape (B, G, G)

            # 4. Calculate the pairwise penalty term
            denominator = torch.abs(reward_diffs_batch) + epsilon_lip_consistency
            pairwise_penalties = (torch.abs(logp_diffs_batch) / denominator) ** 2 # Shape (B, G, G)

            # 5. Sum penalties over all pairs (G, G dimensions) and average over the batch (B dimension)
            consistency_penalty = pairwise_penalties.sum(dim=[1, 2]).mean() # Scalar

            # Log the penalty value (ensure metrics dict is initialized)
            self._metrics["consistency_penalty"].append(self.accelerator.gather_for_metrics(consistency_penalty).mean().item())
        # --- [End New Code] ---

        # --- Final Loss Calculation ---
        final_loss = grpo_loss + lambda_lip_consistency * consistency_penalty


        # --- Metric Logging ---
        # Gather metrics across all processes
        completion_length_gathered = self.accelerator.gather_for_metrics(completion_mask_float.sum(1))
        self._metrics["completion_length"].append(completion_length_gathered.float().mean().item())

        rewards_per_func_gathered = self.accelerator.gather_for_metrics(rewards_per_func) # Shape (Total B*G, F)
        reward_per_func_mean = rewards_per_func_gathered.mean(0) # Shape (F)
        for i, reward_func in enumerate(self.reward_funcs):
             reward_func_name = getattr(reward_func, '__name__', f'reward_func_{i}')
             self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func_mean[i].item())

        rewards_gathered = self.accelerator.gather_for_metrics(rewards) # Shape (Total B*G)
        self._metrics["reward_mean"].append(rewards_gathered.mean().item())

        # std_grouped_rewards was shape (B,1) -> gather needs care
        # Gather the std calculated per batch on each process, then average
        std_gathered = self.accelerator.gather_for_metrics(std_grouped_rewards.squeeze(1)) # Shape (Total B)
        self._metrics["reward_std"].append(std_gathered.mean().item())

        # KL calculation and logging
        # Use the same masked KL calculation as for the loss term
        mean_kl_unreduced = (per_token_kl * completion_mask_float).sum(dim=1) / torch.clamp(sum_completion_mask, min=1e-8) # Shape (B*G)
        mean_kl = mean_kl_unreduced.mean() # Average over B*G on this process
        mean_kl_gathered = self.accelerator.gather_for_metrics(mean_kl) # Gather scalar means
        self._metrics["kl_mean"].append(mean_kl_gathered.mean().item()) # Average the means

        # Log final loss components
        final_loss_gathered = self.accelerator.gather_for_metrics(final_loss)
        self._metrics["loss"].append(final_loss_gathered.mean().item())
        grpo_loss_gathered = self.accelerator.gather_for_metrics(grpo_loss)
        self._metrics["grpo_loss"].append(grpo_loss_gathered.mean().item())

        return final_loss
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()
