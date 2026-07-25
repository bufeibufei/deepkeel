from verification.concurrency_benchmark import run_concurrency_benchmark


def test_shared_runtime_supports_concurrent_independent_turns() -> None:
    report = run_concurrency_benchmark(requests=64, workers=16)

    assert report.succeeded == 64
    assert report.failed == 0
    assert report.success_rate == 1.0
    assert report.throughput_per_second > 0
    assert report.latency_p95_ms < 2_000
