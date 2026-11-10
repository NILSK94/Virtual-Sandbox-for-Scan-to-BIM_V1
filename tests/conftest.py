import pytest

@pytest.fixture(autouse=True)
def isolate_private_credentials(tmp_path,monkeypatch):
    """No test reads or changes the user's real local credential file."""
    monkeypatch.setenv('SCANBIM_SETTINGS_FILE',str(tmp_path/'private-test-settings'/'settings.toml'))
    monkeypatch.delenv('SCANBIM_PROVIDER', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    monkeypatch.delenv('OPENAI_MODEL',raising=False)


@pytest.fixture
def openai_provider(monkeypatch):
    monkeypatch.setenv('SCANBIM_PROVIDER', 'openai')
