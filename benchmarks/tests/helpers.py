"""Explicit test fixtures, not production GPU defaults."""

from sls_bench.config import Target


def sample_targets():
    return [
        Target(
            name="4090", gpu_type="NVIDIA GeForce RTX 4090", expected_memory_gb=24, require_mig=False
        ).model_dump(),
        Target(
            name="mig24",
            gpu_type="NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb",
            expected_memory_gb=24,
            require_mig=True,
        ).model_dump(),
    ]


def seed_test_targets(db):
    for config in sample_targets():
        db.save_config(config)
