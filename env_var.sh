# load cuda129
module load nvidia/25.9

# make sure caches live on $SCRATCH
export UV_CACHE_DIR=$SCRATCH/.cache/uv
export HF_HOME=$SCRATCH/.cache/huggingface

# installation vars
export MAX_JOBS=32
export CC=gcc
export USE_MEGATRON=0
export USE_SGLANG=0

# needed in case embeddings need to be downloaded by hand
export TIKTOKEN_ENCODINGS_BASE=$SCRATCH/data/embeddings

# training vars (adapt as needed)
export TORCH_LOGS=+dynamo
TORCHDYNAMO_VERBOSE=1
export TORCH_COMPILE_DISABLE=1
