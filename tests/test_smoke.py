from claimguard import config


def test_package_imports_and_paths_exist() -> None:
    assert config.ROOT.exists()
    assert config.SEED == 42


def test_placeholder_key_is_not_configured(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", config.PLACEHOLDER_KEY)
    assert config.get_settings().llm_configured is False
    monkeypatch.setenv("GEMINI_API_KEY", "")
    assert config.get_settings().llm_configured is False
    monkeypatch.setenv("GEMINI_API_KEY", "abc123")
    assert config.get_settings().llm_configured is True
