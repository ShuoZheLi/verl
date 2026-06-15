#!/bin/bash
SEEDS=(3542 3968 9489)
CHUNK_SIZES=(128 256)
TEMPLATE="chunk_guidance_eval_mc.sh"

for k in "${CHUNK_SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    name="chunk_eval_k${k}_s${seed}"
    sed \
      -e "s|^#SBATCH --job-name=.*|#SBATCH --job-name=${name}|" \
      -e "s|^#SBATCH --output=.*|#SBATCH --output=slurm-%j_${name}.out|" \
      -e "s|^#SBATCH --error=.*|#SBATCH --error=slurm-%j_${name}.err|" \
      -e "s|^RUN_NAME=.*|RUN_NAME=\"chunk_guidance_eval_7b_mc_critic_k${k}_s${seed}\"|" \
      -e "s|^CHUNK_SIZES=\"128 256\"|CHUNK_SIZES=\"${k}\"|" \
      -e "s|^SEED=\"42\"|SEED=\"${seed}\"|" \
      "$TEMPLATE" | sbatch
    echo "Submitted k=${k}, seed=${seed}"
  done
done
