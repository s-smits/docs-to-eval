"""
Automated test for the Quick Local Test (Qwen) endpoint.
Verifies that the endpoint starts a run without any API key,
progresses to completion, and returns a valid results payload.
"""

import time


def test_qwen_local_endpoint_end_to_end(api_test_client, clear_evaluation_runs):
    # Minimal fictional corpus; include numbers to exercise heuristics
    corpus_text = (
        "The Crystal City has 42 guardians and 73 lanterns. "
        "Its levitation array uses 1,247 crystals to stay afloat."
    )

    # Start the quick local evaluation (no API key required)
    payload = {
        "corpus_text": corpus_text,
        "num_questions": 3,
        "use_fictional": True,
        "run_name": "Qwen Local Test (Automated)",
    }

    resp = api_test_client.post("/api/v1/evaluation/qwen-local", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    run_id = data["run_id"]

    # Poll status until completion (or timeout)
    timeout_s = 10.0
    start = time.time()
    status = None
    while time.time() - start < timeout_s:
        s = api_test_client.get(f"/api/v1/evaluation/{run_id}/status")
        assert s.status_code == 200
        status = s.json().get("status")
        if status in {"completed", "error"}:
            break
        time.sleep(0.05)

    assert status == "completed", f"Run did not complete (status={status})"

    # Fetch results
    r = api_test_client.get(f"/api/v1/evaluation/{run_id}/results")
    assert r.status_code == 200
    result_payload = r.json()

    # Validate top-level structure
    assert result_payload.get("status") == "completed"
    assert "results" in result_payload

    # Validate final results payload
    results = result_payload["results"]
    assert results.get("evaluation_type") == "qwen_local"
    assert results.get("model") == "Simulated Qwen (Local)"

    # System capabilities should indicate no API required
    caps = results.get("system_capabilities", {})
    assert caps.get("no_api_required") is True

    # Aggregate metrics should be well-formed
    metrics = results.get("aggregate_metrics", {})
    assert metrics.get("num_questions", 0) >= 1
    assert 0.0 <= metrics.get("mean_score", 0.0) <= 1.0

