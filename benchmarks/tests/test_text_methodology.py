import json
import random
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from sls_bench import worker
from sls_bench.config import Text, defaults
from sls_bench.resolve import resolve

from .test_corpus import CharacterTokenizer


class ChatTokenizer(CharacterTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True
        assert enable_thinking is False
        return "<user>" + messages[0]["content"] + "</user><assistant><think></think>"


def mixed_config():
    config = next(c for c in defaults() if c["name"] == "qwen8b-mixed")
    config.update(
        prompt_mode="chat_template",
        enable_thinking=False,
        concurrency=[1],
        dataset_revision="d" * 40,
        requests=8,
        repeats=2,
    )
    return config


def test_chat_template_and_thinking_control_preserve_exact_total_lengths(tmp_path, monkeypatch):
    tok = ChatTokenizer()
    for length in (512, 8192, 15360, 31744):
        prompt = worker.exact_prompt(
            tok,
            length,
            random.Random(42),
            render=lambda text: worker.render_chat({"enable_thinking": False}, tok, text),
        )
        assert len(tok(prompt).input_ids) == length
        assert prompt.endswith("</assistant>") is False
        assert prompt.endswith("</user><assistant><think></think>")
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "REPORT", {"trials": []})
    cfg = Text(name="chat", prompt_mode="chat_template", enable_thinking=False).model_dump()
    path = worker.corpus(cfg, tok, 4, 42, "prompts.jsonl")
    assert all(
        len(tok(json.loads(line)["prompt"]).input_ids) == 512 for line in path.read_text().splitlines()
    )
    assert worker.REPORT["datasets"][0]["enable_thinking"] is False


def test_pinned_mixed_corpus_has_variable_lengths_and_forced_output_checks(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    data = [
        {
            "id": str(i),
            "conversations": [
                {"from": "human", "value": "a" * (10 + i)},
                {"from": "gpt", "value": "b" * (20 + 2 * i)},
            ],
        }
        for i in range(30)
    ]
    data.append(
        {"conversations": [{"from": "human", "value": "a" * 2000}, {"from": "gpt", "value": "b" * 20}]}
    )
    source.write_text(json.dumps(data))
    download = Mock(return_value=str(source))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "REPORT", {"trials": []})
    cfg, tok = mixed_config(), ChatTokenizer()
    path = worker.corpus(cfg, tok, 8, 42, "mixed.json")
    rows = json.loads(path.read_text())
    assert len(rows) == 8
    assert len({len(r["conversations"][0]["value"]) for r in rows}) > 1
    assert download.call_args.kwargs["revision"] == "d" * 40
    assert worker.REPORT["datasets"][0]["source_sha256"]
    requests = [
        {
            "input_tokens": len(tok(e["conversations"][0]["value"]).input_ids),
            "output_tokens": len(tok(e["conversations"][1]["value"]).input_ids),
            "error": "",
        }
        for e in reversed(rows)
    ]
    assert worker.lengths_match(cfg, tok, path, requests)
    requests[0]["output_tokens"] -= 1
    assert not worker.lengths_match(cfg, tok, path, requests)
    args = worker.bench_command(cfg, "/model", 64, "trial.json", 8, 42, dataset_path=path, request_rate=0.5)
    assert "--max-concurrency" not in args  # No client queue that hides open-loop overload.
    assert "--ignore-eos" in args and "--no-oversample" in args
    assert args[args.index("--request-rate") + 1] == "0.5"
    assert args[args.index("--burstiness") + 1] == "0.5"
    assert args[args.index("--dataset-name") + 1] == "sharegpt"
    assert "--sharegpt-output-len" not in args  # Preserve variable reference output lengths.
    assert "--custom-output-len" not in args
    warmup = worker.bench_command(cfg, "/model", 1, "warm.json", 2, 42, dataset_path=path)
    assert warmup[warmup.index("--request-rate") + 1] == "inf"
    assert warmup[warmup.index("--max-concurrency") + 1] == "1"


@pytest.mark.parametrize("probe_rate", [100, 10])
def test_mixed_worker_records_rate_not_fake_concurrency_and_keeps_tp(tmp_path, monkeypatch, probe_rate):
    config = {
        **mixed_config(),
        "config_id": 1,
        "model_revision": "model-sha",
        "request_rates": [0.5, 1],
        "saturation_min_output_tok_s": 25,
    }
    server = Mock()
    server.poll.return_value = None
    starter = Mock(return_value=server)
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "REPORT", {"trials": [], "phases": []})
    monkeypatch.setattr(worker, "start", starter)
    monkeypatch.setattr(worker, "ready", lambda *a: None)
    monkeypatch.setattr(worker, "stop", lambda *a: None)
    monkeypatch.setattr(worker.time, "sleep", lambda *a: None)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: ChatTokenizer())),
    )
    monkeypatch.setattr(worker, "corpus", lambda *a: tmp_path / "mixed.json")
    monkeypatch.setattr(worker, "lengths_match", lambda *a: True)

    class Telemetry:
        def __init__(self, *a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(worker, "Telemetry", Telemetry)

    def command(args, *a, **kwargs):
        count = int(args[args.index("--num-prompts") + 1])
        filename = args[args.index("--result-filename") + 1]
        (tmp_path / filename).write_text(
            json.dumps(
                {
                    "completed": count,
                    "output_throughput": probe_rate,
                    "input_lens": [10] * count,
                    "output_lens": [20] * count,
                }
            )
        )

    monkeypatch.setattr(worker, "command", command)
    if probe_rate < 25:
        with pytest.raises(worker.LowThroughputStop):
            worker.fixed_text_workload(config, {"gpu_count": 2}, {(config["model"], "model-sha"): "/model"})
        assert len(worker.REPORT["trials"]) == 1
        assert worker.REPORT["trials"][0]["status"] == "skipped_low_throughput"
        assert not list(tmp_path.glob("*-r0-*"))  # Never started a measured request batch.
        return
    worker.fixed_text_workload(config, {"gpu_count": 2}, {(config["model"], "model-sha"): "/model"})
    assert len(list(tmp_path.glob("*.json.gz"))) == 4
    args = starter.call_args.args[0]
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert args[args.index("--max-num-seqs") + 1] == "64"
    assert "--no-enable-prefix-caching" in args
    trials = worker.REPORT["trials"]
    assert len(trials) == 4
    assert [t["request_rate"] for t in trials] == [0.5, 1, 0.5, 1]
    assert all(t["load_mode"] == "open_loop_request_rate" and t["concurrency"] == 0 for t in trials)
    assert all(t["client_concurrency_limit"] is None and t["status"] == "completed" for t in trials)


def test_live_throughput_guard_has_grace_and_stops_sustained_slowdown(monkeypatch):
    monkeypatch.setattr(worker.time, "monotonic", lambda: 0)
    monkeypatch.setattr(worker, "REPORT", {"trials": []})
    monkeypatch.setattr(worker, "checkpoint", lambda: None)
    watch = worker.ThroughputWatch({"config_id": 1, "saturation_min_output_tok_s": 25})
    for now in range(0, 180, 10):
        watch.observe(now, now * 10)
    with pytest.raises(worker.LowThroughputStop, match="below floor"):
        watch.observe(180, 1800)
    assert worker.REPORT["throughput_probes"][0]["phase"] == "live_generation"
    healthy = worker.ThroughputWatch({"config_id": 1, "saturation_min_output_tok_s": 25}, prior_best=100)
    for now in range(0, 250, 10):
        healthy.observe(now, now * 60)


def test_live_guard_reads_real_token_counters_and_fails_closed(monkeypatch):
    response = Mock()
    response.read.return_value = b'# HELP example\nvllm:generation_tokens_total{engine="0"} 50.0\nvllm:generation_tokens_total{engine="1"} 25.0\nvllm:generation_tokens_created{engine="0"} 999.0\n'
    monkeypatch.setattr(worker.urllib.request, "urlopen", lambda *a, **k: nullcontext(response))
    watch = worker.ThroughputWatch({"config_id": 1, "saturation_min_output_tok_s": 25})
    watch()
    assert watch.samples[0][1] == 75

    def unavailable(*a, **k):
        raise OSError("metrics unavailable")

    monkeypatch.setattr(worker.urllib.request, "urlopen", unavailable)
    for _ in range(8):
        watch()
    with pytest.raises(RuntimeError, match="refusing an unguarded long trial"):
        watch()


def test_dataset_commit_is_resolved_before_allocation(tmp_path):
    calls = []

    def respond(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"sha": ("d" if "/datasets/" in request.url.path else "m") * 40})

    config = mixed_config()
    config["dataset_revision"] = "main"
    config["name"] = "mixed"
    target = {"spec": json.dumps({"image": "example/runtime@sha256:fixed", "gpu_count": 1})}
    client = httpx.Client(transport=httpx.MockTransport(respond))
    result = resolve(target, [{"id": 1, "spec": json.dumps(config)}], client)
    assert result["workloads"][0]["model_revision"] == "m" * 40
    assert result["workloads"][0]["dataset_revision"] == "d" * 40
    assert len(calls) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"saturation_max_concurrency": 1024},
        {"concurrency": [513], "requests": 1024},
        {"max_num_seqs": 513},
        {"enable_thinking": False},
        {"prompt_mode": "chat_template"},
    ],
)
def test_global_cap_and_explicit_thinking_validation(kwargs):
    with pytest.raises(ValidationError):
        Text(name="invalid", **kwargs)


def test_standalone_worker_rejects_old_uncapped_specs_and_unsupported_fp8(monkeypatch):
    config = Text(name="fp8", min_compute_capability=90).model_dump()
    monkeypatch.setattr(worker, "REPORT", {"hardware": {"torch_devices": [{"compute_capability": [8, 9]}]}})
    with pytest.raises(RuntimeError, match="Unsupported GPU"):
        worker.text_workload(config, {}, {})
    config["saturation_max_concurrency"] = 1024
    with pytest.raises(ValueError, match="cap is 512"):
        worker.text_workload(config, {}, {})
