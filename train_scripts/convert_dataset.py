import pandas as pd
import re

def extract_boxed_answer(text: str | None):
    if not isinstance(text, str):
        return None
    m = re.search(r'\\boxed\{([^}]*)\}', text)
    return m.group(1).strip() if m else None

def convert(inp, outp, data_source="lighteval/MATH"):
    df = pd.read_parquet(inp)

    df["prompt"] = df["problem"].apply(lambda x: [{"role": "user", "content": x}])
    gt = df["solution"].apply(extract_boxed_answer)

    df = df[gt.notna()].copy()
    df["data_source"] = data_source

    # What VERL reward_manager expects:
    df["reward_model"] = gt.apply(lambda x: {"ground_truth": x})

    # Save only what you need (include data_source + reward_model!)
    df[["prompt", "reward_model", "data_source"]].to_parquet(outp, index=False)

convert(
    "/data/shuozhe/saved_dataset/math-500/test-00000-of-00001.parquet",
    "/data/shuozhe/saved_dataset/math-500/test-00000-of-00001_verl.parquet",
)