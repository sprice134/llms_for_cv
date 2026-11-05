#!/usr/bin/env python3
"""Run resumable model inference over an AI4Mat task manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("AI4MAT_REQUEST_TIMEOUT", os.environ.get("REQUEST_TIMEOUT", "300")))
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai4mat_common import REPO_ROOT, iter_jsonl, load_yaml, slugify, write_json_atomic


DEFAULT_EXPERIMENTS = "configs/experiments.yaml"
DEFAULT_MODELS = "configs/models.yaml"
DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=False)
    parser.add_argument("--experiments-config", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--models-config", default=DEFAULT_MODELS)
    parser.add_argument("--model-label", required=False)
    parser.add_argument("--run-id", default="run1")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-id", action="append", default=None)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help=(
            "Maximum observed total tokens for this model/run/task selection. "
            "Existing successful outputs are counted when resuming."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help=(
            "Override the model config max_tokens value for this run. For OpenAI "
            "models this is sent as max_completion_tokens."
        ),
    )
    parser.add_argument(
        "--service-tier",
        choices=("auto", "default", "flex", "scale", "priority"),
        default=None,
        help=(
            "Optional OpenAI service_tier value. When omitted, no service_tier "
            "is sent and the API default behavior is used."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of concurrent inference requests. The token budget is enforced "
            "between completed requests, so worker mode may exceed it by in-flight calls."
        ),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_snapshot(command: list[str], timeout: float = 8.0) -> dict[str, Any]:
    executable = shutil.which(command[0])
    snapshot: dict[str, Any] = {"command": command, "available": executable is not None}
    if executable is None:
        return snapshot
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic only.
        snapshot["error"] = str(exc)
        return snapshot
    snapshot.update(
        {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )
    return snapshot


def backend_flavor_for(provider: str, base_url: str, model: dict[str, Any]) -> str:
    configured = model.get("backend_flavor")
    if configured:
        return str(configured)
    if provider == "lmstudio" and base_url.rstrip("/") != DEFAULT_LMSTUDIO_BASE_URL:
        return "vllm"
    return provider


def merge_nested_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_nested_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def collect_environment(
    provider: str,
    base_url: str,
    model_id: str,
    backend_flavor: str,
) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "captured_at": utc_now(),
        "provider": provider,
        "backend_flavor": backend_flavor,
        "base_url": base_url,
        "model_id": model_id,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if provider == "lmstudio":
        environment["lmstudio"] = {
            "lms_which": shutil.which("lms"),
            "lms_version": command_snapshot(["lms", "--version"]),
            "server_status": command_snapshot(["lms", "server", "status"]),
            "runtime_ls": command_snapshot(["lms", "runtime", "ls"]),
            "loaded_models": command_snapshot(["lms", "ps"]),
        }
    return environment


def load_model_entry(models_config: dict[str, Any], label: str) -> dict[str, Any]:
    for model in models_config.get("models", []):
        if model.get("label") == label:
            defaults = dict(models_config.get("defaults", {}))
            merged = {**defaults, **model}
            merged["label"] = label
            return merged
    raise SystemExit(f"Unknown model label {label!r}. Use --list-models.")


def output_complete(path: Path, retry_errors: bool) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    status = payload.get("status")
    if status == "success":
        return True
    if status == "error":
        return not retry_errors
    return False


def output_token_count(payload: dict[str, Any]) -> int:
    usage = payload.get("usage") or {}
    total = usage.get("total_tokens")
    if isinstance(total, (int, float)):
        return max(0, int(total))

    inferred_total = 0
    for key in ("input_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            inferred_total += int(value)
    return max(0, inferred_total)


def read_existing_success(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "success":
        return None
    return payload


def existing_token_count(out_dir: Path, tasks: list[dict[str, Any]], *, force: bool) -> int:
    if force:
        return 0
    total = 0
    for task in tasks:
        payload = read_existing_success(out_dir / f"{slugify(task['task_id'])}.json")
        if payload is not None:
            total += output_token_count(payload)
    return total


class ChatClient:
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Request failed: {exc.reason}") from exc


def build_system_prompt(model: dict[str, Any]) -> str:
    system_prompt = str(model.get("system_prompt", ""))
    reasoning_level = model.get("reasoning_level")
    if reasoning_level:
        system_prompt = f"Reasoning: {reasoning_level}\n{system_prompt}"
    return system_prompt


def chat_completion(
    client: ChatClient,
    *,
    provider: str,
    backend_flavor: str,
    model: dict[str, Any],
    prompt: str,
    service_tier: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_meta: dict[str, Any] = {
        "temperature_sent": False,
        "temperature_retry": False,
        "stop_sent": False,
        "service_tier_sent": service_tier,
        "extra_body_keys": [],
        "backend_flavor": backend_flavor,
    }
    payload: dict[str, Any] = {
        "model": model["model_id"],
        "messages": [
            {"role": "system", "content": build_system_prompt(model)},
            {"role": "user", "content": prompt},
        ],
    }
    max_tokens = model.get("max_tokens")
    temperature = model.get("temperature")
    stop_sequences = model.get("stop_sequences")
    request_extra_body = model.get("request_extra_body") or {}
    if backend_flavor == "vllm":
        if model.get("vllm_stop_sequences") is not None:
            stop_sequences = model.get("vllm_stop_sequences")
        request_extra_body = merge_nested_dicts(
            request_extra_body,
            model.get("vllm_request_extra_body") or {},
        )

    if provider == "openai":
        if service_tier is not None:
            payload["service_tier"] = service_tier
        if model.get("reasoning_effort") is not None:
            payload["reasoning_effort"] = model["reasoning_effort"]
        if max_tokens is not None:
            payload["max_completion_tokens"] = int(max_tokens)
        if temperature is not None:
            payload["temperature"] = float(temperature)
            request_meta["temperature_sent"] = True
    else:
        payload["temperature"] = float(temperature if temperature is not None else 0.0)
        payload["max_tokens"] = int(max_tokens or 512)
        request_meta["temperature_sent"] = True
        if stop_sequences:
            payload["stop"] = stop_sequences
            request_meta["stop_sent"] = True
        if request_extra_body:
            payload.update(request_extra_body)
            request_meta["extra_body_keys"] = sorted(request_extra_body.keys())

    try:
        return client.request("/chat/completions", payload), request_meta
    except RuntimeError as exc:
        if provider == "openai" and "temperature" in str(exc).lower() and "temperature" in payload:
            payload.pop("temperature", None)
            request_meta["temperature_retry"] = True
            return client.request("/chat/completions", payload), request_meta
        raise


def normalize_response(raw: dict[str, Any]) -> dict[str, Any]:
    choices = raw.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    usage = raw.get("usage") or {}
    return {
        "response": (message.get("content") or "").strip(),
        "finish_reason": choice.get("finish_reason"),
        "service_tier": raw.get("service_tier"),
        "usage": {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": (
                usage.get("completion_tokens_details", {}) or {}
            ).get("reasoning_tokens"),
        },
        "raw_response": raw,
    }


def task_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def output_dir_for(config: dict[str, Any], model_label: str, run_id: str) -> Path:
    return REPO_ROOT / config["paths"]["raw_results_dir"] / slugify(model_label) / slugify(run_id)


def run_one_task(
    *,
    task: dict[str, Any],
    args: argparse.Namespace,
    model: dict[str, Any],
    provider: str,
    backend_flavor: str,
    client: ChatClient,
) -> dict[str, Any]:
    attempts = int(model.get("retries", 2)) + 1
    payload: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        start = time.perf_counter()
        try:
            raw, request_meta = chat_completion(
                client,
                provider=provider,
                backend_flavor=backend_flavor,
                model=model,
                prompt=task["prompt"],
                service_tier=args.service_tier,
            )
            runtime = time.perf_counter() - start
            normalized = normalize_response(raw)
            payload = {
                "status": "success",
                "created_at": utc_now(),
                "model_label": args.model_label,
                "provider": provider,
                "backend_flavor": backend_flavor,
                "model_id": model["model_id"],
                "run_id": args.run_id,
                "task_id": task["task_id"],
                "task_suite": task.get("suite"),
                "answer_type": task.get("answer_type"),
                "prompt_sha256_16": task_hash(task["prompt"]),
                "request_metadata": request_meta,
                "runtime": {"elapsed_seconds": runtime},
                **normalized,
            }
            break
        except Exception as exc:  # noqa: BLE001 - persisted run diagnostics.
            runtime = time.perf_counter() - start
            payload = {
                "status": "error",
                "created_at": utc_now(),
                "model_label": args.model_label,
                "provider": provider,
                "backend_flavor": backend_flavor,
                "model_id": model["model_id"],
                "run_id": args.run_id,
                "task_id": task["task_id"],
                "task_suite": task.get("suite"),
                "answer_type": task.get("answer_type"),
                "prompt_sha256_16": task_hash(task["prompt"]),
                "attempt": attempt,
                "runtime": {"elapsed_seconds": runtime},
                "error": str(exc),
            }
            if attempt < attempts:
                time.sleep(min(2.0 * attempt, 10.0))

    assert payload is not None
    return payload


def print_task_result(
    *,
    index: int,
    total_tasks: int,
    task: dict[str, Any],
    payload: dict[str, Any],
    tokens_consumed: int,
    token_budget: int | None,
) -> None:
    budget_text = ""
    if token_budget is not None:
        budget_text = f" tokens={tokens_consumed:,}/{token_budget:,}"
    print(
        f"{index}/{total_tasks} {task['task_id']} -> {payload['status']} "
        f"({payload['runtime']['elapsed_seconds']:.2f}s){budget_text}"
    )


def main() -> None:
    args = parse_args()
    experiments = load_yaml(args.experiments_config)
    models_config = load_yaml(args.models_config)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    if args.list_models:
        for model in models_config.get("models", []):
            marker = " (optional)" if model.get("optional") else ""
            print(f"{model['label']}: {model['provider']} {model['model_id']}{marker}")
        return

    if not args.model_label:
        raise SystemExit("--model-label is required unless --list-models is used")
    if not args.manifest:
        image_limit = int(experiments["image_policy"]["default_limit"])
        args.manifest = f"{experiments['paths']['manifests_dir']}/tasks_image{image_limit}.jsonl"

    model = load_model_entry(models_config, args.model_label)
    if args.max_output_tokens is not None:
        model["max_tokens"] = int(args.max_output_tokens)
    provider = model["provider"]
    if args.service_tier is not None and provider != "openai":
        raise SystemExit("--service-tier is only supported for provider openai")
    base_url = args.base_url
    if base_url is None:
        base_url = DEFAULT_OPENAI_BASE_URL if provider == "openai" else DEFAULT_LMSTUDIO_BASE_URL
    backend_flavor = backend_flavor_for(provider, base_url, model)

    tasks = list(iter_jsonl(args.manifest))
    if args.task_id:
        wanted = set(args.task_id)
        tasks = [task for task in tasks if task["task_id"] in wanted]
    if args.start_index > 1:
        tasks = tasks[args.start_index - 1 :]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    if args.dry_run:
        print(f"Would run {len(tasks):,} tasks for {args.model_label}")
        print(f"Provider/backend: {provider}/{backend_flavor}")
        if args.token_budget is not None:
            print(f"Token budget: {args.token_budget:,}")
        if args.service_tier is not None:
            print(f"OpenAI service_tier: {args.service_tier}")
        for task in tasks[:3]:
            print(f"\n[{task['task_id']}]\n{task['prompt'][:1200]}")
        return

    api_key = args.api_key
    if api_key is None:
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        else:
            api_key = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
    if not api_key:
        raise SystemExit(f"Missing API key for provider {provider}")

    client = ChatClient(base_url, api_key, float(model.get("timeout_seconds", 300)))
    out_dir = output_dir_for(experiments, args.model_label, args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(provider, base_url, model["model_id"], backend_flavor)
    tokens_consumed = existing_token_count(out_dir, tasks, force=args.force)
    environment["token_budget"] = {
        "requested_total_tokens": args.token_budget,
        "existing_success_tokens_counted": tokens_consumed,
        "counts_existing_success_outputs": not args.force,
    }
    environment["openai_service_tier"] = args.service_tier
    write_json_atomic(out_dir / "_run_environment.json", environment)

    completed = 0
    skipped = 0
    budget_stop = False
    runnable: list[tuple[int, dict[str, Any], Path]] = []
    for index, task in enumerate(tasks, start=1):
        if args.token_budget is not None and tokens_consumed >= args.token_budget:
            budget_stop = True
            print(
                "Token budget reached before next request: "
                f"{tokens_consumed:,}/{args.token_budget:,} total tokens."
            )
            break

        output_path = out_dir / f"{slugify(task['task_id'])}.json"
        if not args.force and output_complete(output_path, args.retry_errors):
            skipped += 1
            continue
        runnable.append((index, task, output_path))

    if args.workers == 1:
        for index, task, output_path in runnable:
            if args.token_budget is not None and tokens_consumed >= args.token_budget:
                budget_stop = True
                print(
                    "Token budget reached before next request: "
                    f"{tokens_consumed:,}/{args.token_budget:,} total tokens."
                )
                break
            payload = run_one_task(
                task=task,
                args=args,
                model=model,
                provider=provider,
                backend_flavor=backend_flavor,
                client=client,
            )
            write_json_atomic(output_path, payload)
            if payload.get("status") == "success":
                tokens_consumed += output_token_count(payload)
            completed += 1
            print_task_result(
                index=index,
                total_tasks=len(tasks),
                task=task,
                payload=payload,
                tokens_consumed=tokens_consumed,
                token_budget=args.token_budget,
            )
    else:
        pending = iter(runnable)
        active: dict[Future[dict[str, Any]], tuple[int, dict[str, Any], Path]] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            while True:
                while len(active) < args.workers:
                    if args.token_budget is not None and tokens_consumed >= args.token_budget:
                        budget_stop = True
                        break
                    try:
                        index, task, output_path = next(pending)
                    except StopIteration:
                        break
                    future = executor.submit(
                        run_one_task,
                        task=task,
                        args=args,
                        model=model,
                        provider=provider,
                        backend_flavor=backend_flavor,
                        client=client,
                    )
                    active[future] = (index, task, output_path)
                if not active:
                    break
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    index, task, output_path = active.pop(future)
                    payload = future.result()
                    write_json_atomic(output_path, payload)
                    if payload.get("status") == "success":
                        tokens_consumed += output_token_count(payload)
                    completed += 1
                    print_task_result(
                        index=index,
                        total_tasks=len(tasks),
                        task=task,
                        payload=payload,
                        tokens_consumed=tokens_consumed,
                        token_budget=args.token_budget,
                    )
                if budget_stop:
                    break

    print(
        f"Done. Completed/wrote {completed:,}; skipped {skipped:,}; "
        f"tokens counted {tokens_consumed:,}."
        + (" Stopped at token budget." if budget_stop else "")
        + f" Output: {out_dir}"
    )


if __name__ == "__main__":
    main()
