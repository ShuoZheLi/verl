# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


import os
import uuid
from collections import defaultdict
from contextlib import nullcontext
from functools import partial

import numpy as np
from tensordict.tensorclass import NonTensorData

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging

import hydra
import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint import CheckpointHandler
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.distributed import destroy_global_process_group
from verl.utils.logger import log_with_rank
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.reward_score import default_compute_score
from verl.utils.tracking import Tracking
from verl.workers.engine_workers import TrainingWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class SFTTrainer:
    def __init__(
        self,
        config,
    ):
        self.config = config

        log_gpu_memory_usage(f"rank {torch.distributed.get_rank()}: Before SFTTrainer init", logger=logger)

        self.rank = torch.distributed.get_rank()

        self._build_config()
        self.eval_methods = normalize_eval_methods(self.config.trainer.get("eval_method", "loss"))
        self._build_dataset()

        self._build_engine()

        self._build_dataloader()

        self._init_engine()

        self._build_ckpt_handler()

        # Initialize resume-related variables
        self.resume_global_step = self.ckpt_handler.load_checkpoint()

        self.device_name = self.config.trainer.device

        if self.rank == 0:
            print(self.config)

        log_gpu_memory_usage(f"rank {self.rank}: After SFTTrainer init", logger=logger)

    def _build_ckpt_handler(self):
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)

        self.ckpt_handler = CheckpointHandler(
            engine=self.engine,
            train_dataloader=self.train_dataloader,
            default_local_dir=self.config.trainer.default_local_dir,
            max_ckpt_to_keep=max_ckpt_to_keep,
            default_hdfs_dir=default_hdfs_dir,
            resume_mode=resume_mode,
            resume_from_path=resume_from_path,
        )

    def _build_config(self):
        from verl.utils.config import omega_conf_to_dataclass

        self.model_config = omega_conf_to_dataclass(self.config.model)
        self.engine_config = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)
        self.profiler_config = omega_conf_to_dataclass(self.config.profiler)

        # check profile interval
        self.profiler_interval = self.config.trainer.profile_interval
        self._validate_profiler_interval()

    def _validate_profiler_interval(self):
        assert len(self.profiler_interval) == 2
        self.start_profile_step = self.profiler_interval[0]
        self.end_profile_step = self.profiler_interval[1]
        assert self.end_profile_step >= self.start_profile_step
        if self.start_profile_step < 0:
            assert self.end_profile_step < 0

    def _build_engine(self):
        from verl.workers.engine_workers import TrainingWorkerConfig
        from verl.workers.utils.losses import sft_loss

        self.loss_fn = partial(sft_loss, config=None)

        config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

        self.training_client = TrainingWorker(config=config)
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        # Note that in SPMD world, this abstraction has to break
        self.engine = self.training_client.engine

    def _init_engine(self):
        # patch optimizer config
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps

        self.steps_per_epoch = len(self.train_dataloader)

        # manage save and test frequency
        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch

        self.test_freq = self._normalize_frequency(self.config.trainer.test_freq)
        self.loss_test_freq = self._normalize_frequency(self.config.trainer.get("loss_test_freq", None))
        self.generation_reward_test_freq = self._normalize_frequency(
            self.config.trainer.generation_eval.get("test_freq", None)
        )

        self.training_client.reset()

    def _normalize_frequency(self, frequency):
        if frequency == "after_each_epoch":
            return self.steps_per_epoch
        return frequency

    def _get_eval_frequency(self, method):
        if method == "loss" and self.loss_test_freq is not None:
            return self.loss_test_freq
        if method == "generation_reward" and self.generation_reward_test_freq is not None:
            return self.generation_reward_test_freq
        return self.test_freq

    def _should_validate_method(self, method, global_step, is_last_step):
        if method == "loss" and self.val_dataloader is None:
            return False
        if method == "generation_reward" and self.generation_val_dataloader is None:
            return False
        frequency = self._get_eval_frequency(method)
        return is_last_step or (frequency is not None and frequency > 0 and global_step % frequency == 0)

    def _build_dataset(self):
        config = self.config
        tokenizer = self.model_config.tokenizer
        processor = self.model_config.processor
        train_dataset = create_sft_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        loss_val_files = config.data.get("loss_val_files", None) or config.data.val_files
        if loss_val_files and "loss" in self.eval_methods:
            val_dataset = create_sft_dataset(
                loss_val_files,
                config.data,
                tokenizer,
                processor,
                max_samples=config.data.get("val_max_samples", -1),
            )
        else:
            val_dataset = None

        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.generation_val_dataset = None

        generation_val_files = config.data.get("generation_eval_files", None) or config.data.val_files
        if "generation_reward" in self.eval_methods:
            if not generation_val_files:
                raise ValueError(
                    "data.val_files or data.generation_eval_files must be set when "
                    "trainer.eval_method includes 'generation_reward'"
                )
            self.generation_val_dataset = create_generation_eval_dataset(
                generation_val_files,
                config.data,
                tokenizer,
                processor,
                max_samples=config.data.get("val_max_samples", -1),
            )

    def _build_dataloader(self):
        # build dataset
        config = self.config
        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.collate_fn = SFTTensorCollator(config.data.pad_mode)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.config.data.num_workers,
                pin_memory=False,
                drop_last=True,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None

        if self.generation_val_dataset:
            from verl.utils.dataset.rl_dataset import collate_fn as rlhf_collate_fn

            self.generation_val_sampler = DistributedSampler(
                self.generation_val_dataset,
                shuffle=False,
                num_replicas=dp_size,
                rank=dp_rank,
                drop_last=False,
            )
            generation_eval_batch_size = self.config.data.get("generation_eval_batch_size", None)
            if generation_eval_batch_size is None:
                generation_eval_batch_size = self.train_batch_size_per_dp
            self.generation_val_dataloader = StatefulDataLoader(
                dataset=self.generation_val_dataset,
                batch_size=generation_eval_batch_size,
                sampler=self.generation_val_sampler,
                collate_fn=rlhf_collate_fn,
                num_workers=self.config.data.num_workers,
                pin_memory=False,
                drop_last=False,
                pin_memory_device=device_name,
            )
        else:
            self.generation_val_dataloader = None

    def _get_batch_seqlens(self, data):
        # mean over dp group
        is_nested = data["input_ids"].is_nested
        if is_nested:
            batch_seqlens: torch.Tensor = data["input_ids"].offsets().diff()
        else:
            batch_seqlens: torch.Tensor = data["attention_mask"].sum(dim=-1)
        batch_seqlens = batch_seqlens.to(self.device_name)  # (global_bsz // dp)

        dp_group = self.engine.get_data_parallel_group()
        dp_size = self.engine.get_data_parallel_size()

        if dp_size == 1 or dp_group is None:
            return batch_seqlens.tolist()

        output_tensor = torch.empty(
            (batch_seqlens.shape[0] * dp_size,),
            dtype=batch_seqlens.dtype,
            device=self.device_name,
        )  # (global_bsz,)

        torch.distributed.all_gather_into_tensor(
            output_tensor=output_tensor,
            input_tensor=batch_seqlens,
            group=dp_group,
        )

        batch_seqlens = output_tensor.tolist()
        return batch_seqlens

    def _compute_loss_validation_metrics(self, meta_info):
        val_losses = []
        for val_data in self.val_dataloader:
            val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
            output = self.training_client.infer_batch(val_data)

            if self.engine.is_mp_src_rank_with_outputs():
                metrics = tu.get(output, "metrics")
                val_losses.append(metrics["loss"])

        metric = None
        if self.engine.is_mp_src_rank_with_outputs():
            val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))
            dp_group = self.engine.get_data_parallel_group()
            if dp_group is not None:
                torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)
            metric = {"val/loss": val_loss.detach().item()}
        return metric

    def _compute_generation_reward_validation_metrics(self):
        local_eval = evaluate_generation_reward_batches(
            model=self.engine.module,
            tokenizer=self.model_config.tokenizer,
            dataloader=self.generation_val_dataloader,
            device=self.device_name,
            config=self.config,
        )
        gathered_eval = [None for _ in range(self.engine.get_data_parallel_size())]
        dp_group = self.engine.get_data_parallel_group()
        torch.distributed.all_gather_object(gathered_eval, local_eval, group=dp_group)
        if self.engine.get_data_parallel_rank() != 0 or not self.engine.is_mp_src_rank_with_outputs():
            return None

        merged_eval = merge_generation_reward_results(gathered_eval)
        metric = build_generation_reward_metrics(merged_eval)

        sample_count = len(merged_eval["sample_scores"])
        if sample_count > 0:
            metric["val-aux/reward/num_samples"] = sample_count
            metric["val-aux/response_length/mean"] = float(np.mean(merged_eval["sample_response_lengths"]))
            metric["val-aux/response_length/max"] = int(np.max(merged_eval["sample_response_lengths"]))
        return metric

    def _available_eval_methods(self):
        return [method for method in self.eval_methods if self._has_validation_dataloader(method)]

    def _has_validation_dataloader(self, method):
        if method == "loss":
            return self.val_dataloader is not None
        if method == "generation_reward":
            return self.generation_val_dataloader is not None
        raise ValueError(f"Unsupported eval method {method!r}")

    def _validate(self, meta_info, methods=None):
        methods = methods or self._available_eval_methods()
        combined_metric = {}
        for method in methods:
            if method == "loss":
                metric = self._compute_loss_validation_metrics(meta_info)
            elif method == "generation_reward":
                metric = self._compute_generation_reward_validation_metrics()
            else:
                raise ValueError(f"Unsupported eval method {method!r}")
            if metric:
                combined_metric.update(metric)
        return combined_metric or None

    def fit(self):
        is_logging = self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0

        # TODO: add a unified tracking
        if is_logging:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

        if global_step == 0 and self.config.trainer.get("val_before_train", False):
            metric = self._validate(meta_info)
            if is_logging:
                assert metric is not None, "val_before_train=True but no validation dataloader is available"
                tracking.log(data=metric, step=global_step)
                last_valid_metric = metric
            torch.distributed.barrier()

        train_time = 0
        total_tokens = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            aggressive_empty_cache(force_sync=True)
            log_gpu_memory_usage(f"rank {self.rank}: At start of epoch {epoch}", logger=logger)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=not is_logging,
                )
            ):
                global_step += 1

                # construct tensordict
                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                batch_seqlens = self._get_batch_seqlens(data=data)
                # this is necessary. Otherwise, it is interpreted as NonTensorStack
                batch_seqlens_ntd = NonTensorData(batch_seqlens)

                tu.assign_non_tensor(data, update_lr_scheduler=True, global_token_num=batch_seqlens_ntd)

                # start profile in SPMD mode
                if global_step == self.start_profile_step:
                    self.training_client.start_profile()
                # train for on batch
                output = self.training_client.train_batch(data=data)

                if global_step == self.end_profile_step:
                    self.training_client.stop_profile()

                if self.engine.is_mp_src_rank_with_outputs():
                    metrics = tu.get(output, "metrics")

                    # TODO: we can actual accumulate metrics for N steps and perform aggregate metrics
                    for k in ["loss", "grad_norm", "lr", "mfu"]:
                        if k in metrics.keys():
                            value = metrics.pop(k)
                            metrics[f"train/{k}"] = value

                    metrics["train/global_tokens"] = torch.sum(
                        torch.tensor(batch_seqlens, device=self.device_name)
                    ).item()
                    total_tokens += metrics["train/global_tokens"]
                    metrics["train/total_tokens(B)"] = total_tokens / 1e9

                    if self.engine.get_data_parallel_rank() == 0:
                        tracking.log(data=metrics, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_save_step = global_step % self.save_freq == 0

                # early exit or validation step
                methods_to_validate = [
                    method for method in self.eval_methods if self._should_validate_method(method, global_step, is_last_step)
                ]
                if methods_to_validate:
                    metric = self._validate(meta_info, methods=methods_to_validate)
                    if is_logging:
                        assert metric is not None
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or (self.save_freq > 0 and is_save_step):
                    aggressive_empty_cache(force_sync=True)
                    self.ckpt_handler.save_checkpoint(step=global_step)

                if is_last_step:
                    if is_logging:
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return



def normalize_eval_methods(eval_method):
    if eval_method is None:
        eval_method = "loss"
    if isinstance(eval_method, str):
        if eval_method in {"both", "all"}:
            candidates = ["loss", "generation_reward"]
        else:
            candidates = [method.strip() for method in eval_method.split(",") if method.strip()]
    else:
        candidates = list(eval_method)

    supported_methods = {"loss", "generation_reward"}
    methods = []
    for method in candidates:
        if method not in supported_methods:
            raise ValueError(
                f"Unsupported trainer.eval_method entry {method!r}. "
                "Expected 'loss', 'generation_reward', 'both', or a list of supported methods."
            )
        if method not in methods:
            methods.append(method)
    if not methods:
        raise ValueError("trainer.eval_method must enable at least one evaluation method")
    return tuple(methods)

def run_sft(config):
    from verl.utils.distributed import initialize_global_process_group

    initialize_global_process_group()
    trainer = SFTTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, processor, max_samples=-1):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_object

        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
    else:
        # Default to multi-turn dataset
        dataset_cls = MultiTurnSFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(
        parquet_files=data_paths, tokenizer=tokenizer, config=data_config, processor=processor, max_samples=max_samples
    )
    return dataset


def create_generation_eval_dataset(data_paths, data_config, tokenizer, processor, max_samples=-1):
    from verl.utils.dataset.rl_dataset import RLHFDataset

    eval_data_config = OmegaConf.create(OmegaConf.to_container(data_config, resolve=True))
    eval_data_config.prompt_key = eval_data_config.get("generation_eval_prompt_key", "prompt")
    eval_data_config.max_prompt_length = eval_data_config.get("generation_eval_max_prompt_length", 2048)
    eval_data_config.filter_overlong_prompts = eval_data_config.get("generation_eval_filter_overlong_prompts", False)
    eval_data_config.filter_overlong_prompts_workers = eval_data_config.get("filter_overlong_prompts_workers", 1)
    eval_data_config.shuffle = False
    eval_data_config.return_raw_chat = True
    eval_data_config.return_full_prompt = False
    eval_data_config.return_raw_input_ids = False
    eval_data_config.truncation = eval_data_config.get("generation_eval_truncation", "error")
    return RLHFDataset(
        data_files=data_paths,
        tokenizer=tokenizer,
        config=eval_data_config,
        processor=processor,
        max_samples=max_samples,
    )


def extract_prompt_texts(raw_prompts, tokenizer, apply_chat_template_kwargs=None):
    apply_chat_template_kwargs = apply_chat_template_kwargs or {}
    prompt_texts = []
    for raw_prompt in raw_prompts:
        messages = raw_prompt.tolist() if hasattr(raw_prompt, "tolist") else raw_prompt
        prompt_texts.append(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs,
            )
        )
    return prompt_texts




def get_torch_dtype(dtype_name):
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported generation dtype {dtype_name!r}")
    return dtype_map[dtype_name]


def get_generation_autocast_dtype(model, generation_config):
    configured_dtype = generation_config.get("dtype", None)
    if configured_dtype is not None:
        return get_torch_dtype(configured_dtype)
    try:
        param_dtype = next(parameter.dtype for parameter in model.parameters() if parameter.is_floating_point())
    except StopIteration:
        return torch.bfloat16
    if param_dtype in {torch.bfloat16, torch.float16}:
        return param_dtype
    return torch.bfloat16


def get_fsdp_full_param_context(model):
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        return nullcontext()

    if isinstance(model, FSDP):
        return FSDP.summon_full_params(model, recurse=False, writeback=False)
    return nullcontext()


def generate_texts_from_prompts(model, tokenizer, prompt_texts, device, generation_config):
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = previous_padding_side
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[-1]
    generate_kwargs = {
        "max_new_tokens": generation_config.get("max_new_tokens", 2048),
        "do_sample": generation_config.get("do_sample", False),
        "temperature": generation_config.get("temperature", 1.0),
        "top_p": generation_config.get("top_p", 1.0),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    top_k = generation_config.get("top_k", None)
    if top_k is not None and top_k >= 0:
        generate_kwargs["top_k"] = top_k
    if not generate_kwargs["do_sample"]:
        generate_kwargs.pop("temperature", None)
        generate_kwargs.pop("top_p", None)
        generate_kwargs.pop("top_k", None)

    fsdp_full_param_context = get_fsdp_full_param_context(model)
    with fsdp_full_param_context:
        generate_fn = getattr(model, "generate", None)
        if generate_fn is None:
            wrapped_module = getattr(model, "module", None) or getattr(model, "_fsdp_wrapped_module", None)
            generate_fn = getattr(wrapped_module, "generate", None)
        if generate_fn is None:
            raise AttributeError("The SFT model does not expose a HuggingFace-compatible generate method.")

        autocast_dtype = get_generation_autocast_dtype(model, generation_config)
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, dtype=autocast_dtype):
            output_ids = generate_fn(**inputs, **generate_kwargs)
    response_ids = output_ids[:, prompt_length:]
    output_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    response_lengths = (response_ids != tokenizer.pad_token_id).sum(dim=-1).cpu().tolist()
    return output_texts, response_lengths


def score_generation_outputs(batch, outputs, config):
    scores = []
    reward_extra_infos = defaultdict(list)
    reward_kwargs = config.get("reward_kwargs", {})

    for index, output in enumerate(outputs):
        data_source = batch["data_source"][index]
        reward_model = batch["reward_model"][index]
        extra_info = batch.get("extra_info", np.array([{}] * len(outputs), dtype=object))[index]
        ground_truth = reward_model.get("ground_truth", None)
        score = default_compute_score(
            data_source=data_source,
            solution_str=output,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **reward_kwargs,
        )
        if isinstance(score, dict):
            score_value = float(score.get("score", score.get("reward", 0.0)))
            for key, value in score.items():
                reward_extra_infos[key].append(value)
        else:
            score_value = float(score)
        scores.append(score_value)
        reward_extra_infos["reward"].append(score_value)
        reward_extra_infos["acc"].append(score_value)

    return scores, reward_extra_infos


def evaluate_generation_reward_batches(model, tokenizer, dataloader, device, config):
    result = {
        "data_sources": [],
        "sample_uids": [],
        "reward_extra_infos_dict": defaultdict(list),
        "sample_response_lengths": [],
        "sample_scores": [],
    }
    generation_config = config.trainer.get("generation_eval", {})

    was_training = model.training
    model.eval()
    try:
        for batch in dataloader:
            raw_prompts = batch.get("raw_prompt", batch[config.data.get("generation_eval_prompt_key", "prompt")])
            prompt_texts = extract_prompt_texts(
                raw_prompts,
                tokenizer,
                apply_chat_template_kwargs=config.data.get("apply_chat_template_kwargs", {}),
            )
            repeat_times = int(generation_config.get("n", 1))
            repeated_prompt_texts = [prompt for prompt in prompt_texts for _ in range(repeat_times)]
            outputs, response_lengths = generate_texts_from_prompts(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=repeated_prompt_texts,
                device=device,
                generation_config=generation_config,
            )
            repeated_batch = repeat_non_tensor_batch(batch, repeat_times)
            scores, reward_extra_infos = score_generation_outputs(repeated_batch, outputs, generation_config)

            sample_count = len(outputs)
            if "uid" in repeated_batch:
                uids = repeated_batch["uid"].tolist()
            else:
                uids = [str(uuid.uuid4()) for _ in range(sample_count)]

            result["data_sources"].extend(repeated_batch["data_source"].tolist())
            result["sample_uids"].extend(uids)
            result["sample_scores"].extend(scores)
            result["sample_response_lengths"].extend(response_lengths)
            for key, values in reward_extra_infos.items():
                result["reward_extra_infos_dict"][key].extend(values)
    finally:
        if was_training:
            model.train()

    result["reward_extra_infos_dict"] = dict(result["reward_extra_infos_dict"])
    return result


def repeat_non_tensor_batch(batch, repeat_times):
    if repeat_times == 1:
        return batch
    repeated = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            repeated[key] = value.repeat_interleave(repeat_times, dim=0)
        elif isinstance(value, np.ndarray):
            repeated[key] = np.repeat(value, repeat_times, axis=0)
        else:
            repeated[key] = value
    return repeated


def merge_generation_reward_results(results):
    merged = {
        "data_sources": [],
        "sample_uids": [],
        "reward_extra_infos_dict": defaultdict(list),
        "sample_response_lengths": [],
        "sample_scores": [],
    }
    for result in results:
        if not result:
            continue
        merged["data_sources"].extend(result["data_sources"])
        merged["sample_uids"].extend(result["sample_uids"])
        merged["sample_scores"].extend(result["sample_scores"])
        merged["sample_response_lengths"].extend(result["sample_response_lengths"])
        for key, values in result["reward_extra_infos_dict"].items():
            merged["reward_extra_infos_dict"][key].extend(values)
    merged["reward_extra_infos_dict"] = dict(merged["reward_extra_infos_dict"])
    return merged


def build_generation_reward_metrics(eval_result):
    from verl.trainer.ppo.metric_utils import process_validation_metrics

    if len(eval_result["sample_scores"]) == 0:
        return {}
    data_sources = np.asarray(eval_result["data_sources"], dtype=object)
    metric_tree = process_validation_metrics(
        data_sources,
        eval_result["sample_uids"],
        eval_result["reward_extra_infos_dict"],
    )
    metric = {}
    for data_source, var2metric2val in metric_tree.items():
        core_var = "acc" if "acc" in var2metric2val else "reward"
        for var_name, metric2val in var2metric2val.items():
            n_max = max(int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys())
            for metric_name, metric_val in metric2val.items():
                if (
                    var_name == core_var
                    and any(metric_name.startswith(prefix) for prefix in ["mean", "maj", "best"])
                    and f"@{n_max}" in metric_name
                ):
                    metric_section = "val-core"
                else:
                    metric_section = "val-aux"
                metric[f"{metric_section}/{data_source}/{var_name}/{metric_name}"] = to_builtin_scalar(metric_val)
    return metric


def to_builtin_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
