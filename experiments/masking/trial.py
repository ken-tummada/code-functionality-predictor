import torch

import re
import random
import json
from tqdm import tqdm


def extract_code(text):
    code = text
    if "```python" in text:
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if match:
            code = match.group(1)
    elif "```" in text:
        match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match:
            code = match.group(1)
    return code.strip()


class Trial:
    def __init__(
        self,
        ds,
        base_out_path,
    ):
        self.ds = ds
        self.dump_loc = f"{base_out_path}/model_output.jsonl"
        self.metrics_loc = f"{base_out_path}/metric.json"

    def generate_samples(self, model, tokenizer, p_mask, allow_hint, allowed_gen):
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        for batch in tqdm(self.ds, desc="Generating samples"):
            messages = [{"role": "", "content": batch["code"]}]
            tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                return_tensors="pt",
            )

            offset = 0

            if allow_hint:
                hint_messages = [{"role": "", "content": batch["hint"]}]
                hint_tokens = tokenizer.apply_chat_template(
                    hint_messages,
                    tokenize=True,
                    return_dict=False,
                    return_tensors="pt",
                )

                tokens = torch.cat([hint_tokens, tokens], dim=1)
                offset = hint_tokens.shape[1]

            tokens.to(model.device)

            # FIXME: this is bad, it locks into llama 3.1 8b inst
            offset += 3

            chunk_start = offset
            while chunk_start < tokens.shape[1] - 1:
                # TODO: find better values
                chunk_size = random.randint(15, 30)
                chunk_end = min(chunk_start + chunk_size, tokens.shape[1] - 1)

                if p_mask > random.random():
                    for window in range(chunk_start, chunk_end):
                        inputs = tokens[0][0:window].unsqueeze(0).to(model.device)
                        output = model(inputs, torch.ones(inputs.shape[1]))

                        next_token_logits = output.logits[:, -1, :]
                        next_token_id = torch.argmax(next_token_logits, dim=-1)
                        tokens[0][window] = next_token_id

                chunk_start = chunk_end

            if allowed_gen > 0:
                gen_count = 0
                tokens = tokens[:, : tokens.shape[1] - 1]
                while gen_count < allowed_gen:
                    inputs = tokens[0].unsqueeze(0).to(model.device)
                    output = model(inputs, torch.ones_like(inputs))

                    next_token_logits = output.logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                    tokens = torch.cat([tokens, next_token_id.unsqueeze(0)], dim=1)
                    gen_count += 1

                    if next_token_id.item() in terminators:
                        break

            output = tokenizer.decode(tokens[-1])
            output = re.findall(r"\|>(.*?)<\|", output, re.DOTALL)[-1].strip()
            record = {
                "model_name": model.name,
                "id": batch["id"],
                "generated_code": output,
            }
            with open(self.dump_loc, "a") as f:
                f.write(json.dumps(record) + "\n")

    def eval(self):
        id_to_test = {ex["id"]: ex["tests"] for ex in self.ds}

        errors = 0
        fails = 0
        passes = 0

        with open(self.dump_loc, "r") as f:
            for line in tqdm(f, desc="Evaluating samples"):
                record = json.loads(line.strip())
                task_id = record["id"]
                generated_code = record["generated_code"]

                if task_id not in id_to_test:
                    continue

                code = extract_code(generated_code)
                tests = id_to_test[task_id]

                try:
                    compile(code, "<string>", "exec")
                except SyntaxError:
                    errors += 1
                    continue

                test_module = {}
                try:
                    exec(code, test_module)
                except Exception:
                    errors += 1
                    continue

                try:
                    exec(tests, test_module)
                except AssertionError:
                    fails += 1
                except Exception:
                    fails += 1
                else:
                    passes += 1

        total = errors + fails + passes
        error_rate = errors / total if total > 0 else 0.0
        fail_rate = fails / (total - errors) if total - errors > 0 else 0.0

        metrics = {
            "total": total,
            "errors": errors,
            "fails": fails,
            "passes": passes,
            "error_rate": error_rate,
            "fail_rate": fail_rate,
        }

        with open(self.metrics_loc, "w") as f:
            json.dump(metrics, f, indent=2)

        self.metrics = metrics
        return self.metrics

    def run(self):
        self.generate_samples()
        self.eval()
