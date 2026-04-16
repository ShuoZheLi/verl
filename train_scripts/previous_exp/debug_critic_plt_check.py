import json
from pathlib import Path

out_dir = Path("/nfs/shuozhe/verl/outputs/critic_debug_all")
paths = sorted(out_dir.glob("responses_rank*.jsonl"))
if not paths:
    paths = [out_dir / "responses.jsonl"]

total = correct = missing_ref = 0
for p in paths:
    with p.open() as f:
        for line in f:
            rec = json.loads(line)
            total += 1
            if rec.get("reference") in (None, "", "None"):
                missing_ref += 1
            if rec.get("correct"):
                correct += 1

print("total:", total)
print("correct:", correct)
print("missing_ref:", missing_ref)