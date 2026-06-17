#!/usr/bin/env python3
import argparse
import logging

from transformers import AutoModelForCausalLM

from verl.utils.sparse_update_mask import build_masks_from_model, save_sparse_masks, sparse_mask_metadata

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build sparse-update actor masks from a Hugging Face causal LM.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--mode", default="safe_svd_lowmag")
    parser.add_argument("--rank_k", type=int, default=128)
    parser.add_argument("--alpha_princ", type=float, default=0.5)
    parser.add_argument("--alpha_low", type=float, default=0.5)
    parser.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--exclude_keywords", default="embed,lm_head,norm,layernorm,rmsnorm")
    parser.add_argument("--svd_device", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = {
        "mode": args.mode,
        "rank_k": args.rank_k,
        "alpha_princ": args.alpha_princ,
        "alpha_low": args.alpha_low,
        "target_modules": [item for item in args.target_modules.split(",") if item],
        "exclude_keywords": [item for item in args.exclude_keywords.split(",") if item],
        "apply_to_bias": False,
        "svd_device": args.svd_device,
        "dry_run_log_only": args.dry_run,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
    )
    masks = build_masks_from_model(model, config)
    metadata = sparse_mask_metadata(masks, config, extra={"model_name_or_path": args.model_name_or_path})
    print(f"num_masked_tensors={len(masks)}")
    print(f"linear_trainable_fraction={metadata['linear_trainable_fraction']:.6f}")
    for name, density in metadata["per_param_density"].items():
        print(f"{name}\t{density:.6f}\t{masks[name].numel()}")
    if not args.dry_run:
        save_sparse_masks(args.output_path, masks, metadata)
        print(f"saved {args.output_path}")


if __name__ == "__main__":
    main()
