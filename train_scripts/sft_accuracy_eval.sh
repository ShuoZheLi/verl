set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: sft_accuracy_eval.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Recommended starting point for full SFT of /data/shuozhe/saved_model/Qwen2.5-3B
# on the 7,500-row MetaMathQA/MATH SFT set. This is a small-data fine-tune, so
# use a conservative LR, only a few epochs, and select checkpoints by generation
# accuracy rather than loss alone. Loss can keep improving after math accuracy degrades.
#
# Suggested ablations:
#   Conservative: lr=3e-6, train_batch_size=32, total_epochs=3
#   Main:         lr=5e-6, train_batch_size=32, total_epochs=3
#   Stronger:     lr=1e-5, train_batch_size=64, total_epochs=2
#
# If running on one 24GB GPU, consider overriding:
#   data.train_batch_size=8 optim.lr=3e-6 trainer.total_epochs=3
# Full 3B fine-tuning can still be tight on one GPU depending on sequence lengths.

# Eval options. Defaults run both SFT loss eval and PPO-style generate+reward accuracy eval.
eval_method=${eval_method:-both}
eval_before_train=${eval_before_train:-True}
eval_freq=${eval_freq:--1}
loss_eval_freq=${loss_eval_freq:-50}
generation_eval_freq=${generation_eval_freq:-100}

# loss_eval_files must be SFT messages format; generation_eval_files must be PPO prompt+reward_model format.
loss_eval_files=${loss_eval_files:-/data/shuozhe/saved_dataset/MetaMathQA-math-500/math7500_sft.parquet}
generation_eval_files=${generation_eval_files:-/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}

# Generation accuracy eval is expensive with max_new_tokens=2048, so start with batch size 1.
generation_eval_batch_size=${generation_eval_batch_size:-4}
generation_max_new_tokens=${generation_max_new_tokens:-2048}
generation_do_sample=${generation_do_sample:-False}
generation_temperature=${generation_temperature:-1.0}
generation_top_p=${generation_top_p:-1.0}
generation_top_k=${generation_top_k:-null}
generation_num_samples=${generation_num_samples:-1}
generation_dtype=${generation_dtype:-null}

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.sft_trainer \
    data.train_files=/data/shuozhe/saved_dataset/MetaMathQA-math-500/math7500_sft.parquet \
    data.val_files=/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
    data.loss_val_files=${loss_eval_files} \
    data.generation_eval_files=${generation_eval_files} \
    data.generation_eval_batch_size=${generation_eval_batch_size} \
    data.messages_key=messages \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=4096 \
    data.max_token_len_per_gpu=4096 \
    optim.lr=5e-6 \
    engine=fsdp \
    model.path=/data/shuozhe/saved_model/Qwen2.5-3B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=math_7500 \
    trainer.experiment_name=qwen2.5-3b-math7500-sft \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.test_freq=${eval_freq} \
    trainer.loss_test_freq=${loss_eval_freq} \
    trainer.eval_method=${eval_method} \
    trainer.val_before_train=${eval_before_train} \
    trainer.generation_eval.test_freq=${generation_eval_freq} \
    trainer.generation_eval.max_new_tokens=${generation_max_new_tokens} \
    trainer.generation_eval.do_sample=${generation_do_sample} \
    trainer.generation_eval.temperature=${generation_temperature} \
    trainer.generation_eval.top_p=${generation_top_p} \
    trainer.generation_eval.top_k=${generation_top_k} \
    trainer.generation_eval.n=${generation_num_samples} \
    trainer.generation_eval.dtype=${generation_dtype} \
    trainer.logger='["console","wandb"]' $@
