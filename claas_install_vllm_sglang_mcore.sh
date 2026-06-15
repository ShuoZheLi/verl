#!/bin/bash

source env_vars.sh

uv venv --python=3.12
source .venv/bin/activate

echo "1. install inference frameworks and pytorch they need"
if [ $USE_SGLANG -eq 1 ]; then
    uv pip install "sglang[all]==0.5.2" && uv pip install torch-memory-saver
fi
uv pip install "vllm==0.12.0"

# pip vllm seems to like CPU torch, so forcing cuda torch after
uv pip uninstall torch
uv pip install torch==2.9.0 --torch-backend=cu129

echo "2. install basic packages"
uv pip install "transformers[hf_xet]==4.57.6" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard 

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "uv pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# Install flash-attn-2.8.1 (cxx11abi=False)
uv pip install flash_attn==2.8.3 --no-build-isolation

uv pip install flashinfer-python==0.3.1


if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    uv pip install "onnxscript==0.3.1"
    NVTE_FRAMEWORK=pytorch uv pip install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.6
    uv pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1
fi


echo "5. May need to fix opencv"
uv pip install opencv-python
# uv pip install opencv-fixer && \
#     python -c "from opencv_fixer import AutoFix; AutoFix()"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    uv pip install nvidia-cudnn-cu12==9.10.2.21
fi

# Fix assorted dependencies
uv pip install trl
uv pip install numpy==2.2

echo "Successfully installed all packages"
