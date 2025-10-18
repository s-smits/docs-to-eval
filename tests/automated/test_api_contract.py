"""
Lightweight API contract tests that ensure the public FastAPI routes respond with
the expected payload structure without invoking the heavy agentic pipeline.
"""


def test_health_endpoint(api_test_client):
    response = api_test_client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "active_runs" in payload
    assert "websocket_connections" in payload


def test_default_config_endpoint(api_test_client):
    response = api_test_client.get("/api/v1/config/default")
    assert response.status_code == 200

    config = response.json()
    # Core sections should be present for downstream clients
    for key in ["eval_type", "llm", "generation", "verification", "chunking", "reporting", "system"]:
        assert key in config

    assert config["generation"]["num_questions"] > 0
    assert config["verification"]["similarity_threshold"] <= 1.0
