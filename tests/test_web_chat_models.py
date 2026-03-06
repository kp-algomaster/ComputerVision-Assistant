from fastapi.testclient import TestClient

import cv_agent.web as web
from cv_agent.config import load_config
from cv_agent.web import _is_chat_model_compatible, _select_default_chat_model


def test_chat_model_compatibility_accepts_completion_models():
    assert _is_chat_model_compatible("qwen3.5:9b", ["completion", "tools", "thinking"])


def test_chat_model_compatibility_rejects_embedding_models():
    assert not _is_chat_model_compatible("nomic-embed-text:latest", ["completion"])


def test_select_default_chat_model_prefers_requested_then_configured_then_first():
    models = ["qwen3.5:9b", "gpt-oss:20b"]

    assert _select_default_chat_model(models, "missing:model", "gpt-oss:20b") == "gpt-oss:20b"
    assert _select_default_chat_model(models, "qwen3.5:9b", "missing:model") == "qwen3.5:9b"
    assert _select_default_chat_model(models, "missing:model") == "qwen3.5:9b"
    assert _select_default_chat_model([], "missing:model") == ""


def test_configure_power_creates_env_and_persists_hf_token(monkeypatch, tmp_path):
    persisted: dict[str, str] = {}

    monkeypatch.setattr(web, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        web,
        "_persist_huggingface_token",
        lambda token: persisted.setdefault("token", token) or True,
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    client = TestClient(web.create_app(load_config()))
    response = client.post(
        "/api/powers/huggingface/configure",
        json={"fields": {"HF_TOKEN": "hf_test_token"}},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "updated": ["HF_TOKEN"]}
    assert (tmp_path / ".env").read_text() == "HF_TOKEN=hf_test_token\n"
    assert persisted["token"] == "hf_test_token"
    assert web.os.environ["HF_TOKEN"] == "hf_test_token"
    assert web.os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_test_token"
