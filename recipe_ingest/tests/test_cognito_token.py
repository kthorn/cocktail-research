"""Tests for the Cognito USER_PASSWORD_AUTH token helper."""

import importlib.util
import requests
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).parents[1] / "cognito_token.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cognito_token_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load cognito_token module")
    return importlib.util.module_from_spec(spec), spec


def load():
    module, spec = load_module()
    spec.loader.exec_module(module)
    return module


def test_get_token_returns_access_token():
    module = load()
    with patch("requests.post") as post:
        post.return_value.ok = True
        post.return_value.raise_for_status.return_value = None
        post.return_value.status_code = 200
        post.return_value.json.return_value = {
            "AuthenticationResult": {"AccessToken": "jwt123", "ExpiresIn": 3600}
        }

        token = module.get_token(
            username="kurt", password="secret", client_id="abc", region="us-east-1"
        )

    assert token == "jwt123"
    kwargs = post.call_args.kwargs
    assert "cognito-idp.us-east-1.amazonaws.com" in post.call_args.args[0]
    assert kwargs["json"]["AuthFlow"] == "USER_PASSWORD_AUTH"
    assert kwargs["json"]["ClientId"] == "abc"
    assert kwargs["json"]["AuthParameters"]["USERNAME"] == "kurt"


def test_get_token_raises_clear_error_on_bad_credentials():
    module = load()
    with patch("requests.post") as post:
        post.return_value.ok = False
        post.return_value.status_code = 400
        post.return_value.json.return_value = {
            "__type": "NotAuthorizedException",
            "message": "Incorrect username or password.",
        }
        post.return_value.raise_for_status.side_effect = requests.HTTPError(
            response=post.return_value
        )
        try:
            module.get_token(username="kurt", password="wrong", client_id="abc", region="us-east-1")
        except module.AuthError as exc:
            assert "Incorrect username or password" in str(exc)
            assert "MIXOLOGY_USERNAME" in str(exc)
        else:
            raise AssertionError("expected AuthError")


def test_get_token_missing_credentials_raises_auth_error(monkeypatch):
    module = load()
    for var in ("MIXOLOGY_USERNAME", "MIXOLOGY_PASSWORD"):
        monkeypatch.delenv(var, raising=False)

    try:
        module.get_token_from_env(client_id="abc", region="us-east-1")
    except module.AuthError as exc:
        assert "MIXOLOGY_USERNAME" in str(exc)
    else:
        raise AssertionError("expected AuthError")


def test_get_token_from_env_uses_environment(monkeypatch):
    module = load()
    monkeypatch.setenv("MIXOLOGY_USERNAME", "kurt")
    monkeypatch.setenv("MIXOLOGY_PASSWORD", "secret")
    with patch("requests.post") as post:
        post.return_value.ok = True
        post.return_value.raise_for_status.return_value = None
        post.return_value.status_code = 200
        post.return_value.json.return_value = {
            "AuthenticationResult": {"AccessToken": "jwt123"}
        }

        token = module.get_token_from_env(
            client_id="abc",
            region="us-east-1",
            username=None,
            password=None,
        )

    assert token == "jwt123"
    assert post.call_args.kwargs["json"]["AuthParameters"]["PASSWORD"] == "secret"
