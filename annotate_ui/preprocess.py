import glob
import json

directories = [
    ("outputs/desc-gen-gpt-5-mini", "gpt-5-mini"),
    ("outputs/desc-gen-llama-3-8b", "llama-3.1-8b"),
    ("outputs/desc-gen-sonnet-4.5", "sonnet-4.5"),
]

result = []

for dir, name in directories:
    predictions = []
    for file_name in glob.glob(f"{dir}/*.json"):
        with open(file_name, "r") as f:
            predictions.extend(json.load(f)["preds"])

    for i, pred in enumerate(predictions):
        result.append({"id": f"{name}-{i}", "text": pred})


with open("test.json", "w") as f:
    json.dump(result, f)
