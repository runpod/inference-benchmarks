import json
import random
from types import SimpleNamespace

from sls_bench import worker
from sls_bench.config import Text


class CharacterTokenizer:
    """Includes a BOS token so the generator must account for special tokens too."""

    def __call__(self, text):
        return SimpleNamespace(input_ids=[0] + self.encode(text))

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(c) for c in ids)


def test_exact_lengths_round_trip_and_determinism():
    tokenizer = CharacterTokenizer()
    for length in (512, 8192, 15360, 31744):
        a = worker.exact_prompt(tokenizer, length, random.Random(42))
        b = worker.exact_prompt(tokenizer, length, random.Random(42))
        assert a == b
        assert len(tokenizer(a).input_ids) == length


def test_generated_corpus_hash_and_custom_benchmark_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "REPORT", {"status": "running"})
    config = Text(name="test").model_dump()
    path = worker.corpus(config, CharacterTokenizer(), 4, 42, "dataset.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 4
    assert all(len(CharacterTokenizer()(row["prompt"]).input_ids) == 512 for row in rows)
    assert worker.REPORT["datasets"][0]["sha256"]
    args = worker.bench_command(config, "/model", 1, "trial.json", 4, 42, dataset_path=path)
    assert args[args.index("--dataset-name") + 1] == "custom"
    assert args[args.index("--custom-output-len") + 1] == "256"
    assert "--custom-skip-chat-template" in args
    assert "--random-input-len" not in args


def test_failure_classification_and_cancellation():
    assert worker.failure_kind("CUDA out of memory") == "out_of_memory"
    assert worker.failure_kind("max_model_len larger than maximum KV cache capacity") == "context_capacity"
    assert worker.failure_kind("no kernel image available") == "unsupported_hardware"
    assert not issubclass(worker.WorkerCancelled, Exception)
