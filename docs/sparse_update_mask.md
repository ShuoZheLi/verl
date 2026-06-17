# Sparse Actor Updates

Sparse actor updates can be enabled for the FSDP actor backend without changing PPO/GRPO loss math. The mask is applied around the actor optimizer step: gradients and AdamW state are zeroed for frozen entries, then frozen weights are restored after `optimizer.step()`.

Example Hydra overrides:

```bash
actor_rollout_ref.actor.sparse_update.enabled=true \
actor_rollout_ref.actor.sparse_update.mode=safe_svd_lowmag \
actor_rollout_ref.actor.sparse_update.build_mask_on_init=true \
actor_rollout_ref.actor.sparse_update.rank_k=128 \
actor_rollout_ref.actor.sparse_update.alpha_princ=0.5 \
actor_rollout_ref.actor.sparse_update.alpha_low=0.5 \
actor_rollout_ref.actor.sparse_update.restore_frozen_after_step=true \
actor_rollout_ref.actor.sparse_update.mask_optimizer_state=true \
actor_rollout_ref.actor.sparse_update.verify_frozen_weights=true
```

External mask usage:

```bash
python tools/build_sparse_update_mask.py \
  --model_name_or_path /path/to/base_actor \
  --output_path /path/to/safe_mask.pt \
  --mode safe_svd_lowmag \
  --rank_k 128 \
  --alpha_princ 0.5 \
  --alpha_low 0.5 \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --svd_device cuda

actor_rollout_ref.actor.sparse_update.enabled=true \
actor_rollout_ref.actor.sparse_update.mask_path=/path/to/safe_mask.pt
```

Supported modes are `safe_svd_lowmag`, `non_principal`, `low_magnitude`, `principal`, and `random_same_density`.
