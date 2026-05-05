import json
import os
from pathlib import Path


def record_openai_usage(response, label, model):
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    usage_path = os.environ.get("SREPAIR_USAGE_PATH")
    if not usage_path:
        return

    path = Path(usage_path)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    data = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": [],
    }
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    data["prompt_tokens"] = int(data.get("prompt_tokens", 0)) + prompt_tokens
    data["completion_tokens"] = int(data.get("completion_tokens", 0)) + completion_tokens
    data["total_tokens"] = data["prompt_tokens"] + data["completion_tokens"]
    data.setdefault("calls", []).append(
        {
            "label": label,
            "model": model,
            "prompt_token_count": prompt_tokens,
            "completion_token_count": completion_tokens,
            "total_token_count": prompt_tokens + completion_tokens,
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
