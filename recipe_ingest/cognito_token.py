#!/usr/bin/env python3
"""Cognito access-token helper for the ingest pipeline.

Mints a bearer token via Cognito's USER_PASSWORD_AUTH flow so the pipeline can
call mixology.tools editor endpoints (e.g. ingredient creation) directly.

Credentials come from MIXOLOGY_USERNAME / MIXOLOGY_PASSWORD (or explicit args).
Client ID defaults to the prod app client from cocktaildb's deployment
inventory; override with MIXOLOGY_CLIENT_ID / MIXOLOGY_REGION if they change.
"""

import os

import requests

DEFAULT_CLIENT_ID = "50rcs7t960nm1p0q4n9t56503e"  # prod app client
DEFAULT_REGION = "us-east-1"


class AuthError(Exception):
    """Raised when authentication fails or is not configured."""


def get_token(
    username: str | None,
    password: str | None,
    client_id: str = DEFAULT_CLIENT_ID,
    region: str = DEFAULT_REGION,
) -> str:
    """Get a Cognito access token via USER_PASSWORD_AUTH. AuthError on failure."""
    if not username or not password:
        raise AuthError(
            "No credentials: set MIXOLOGY_USERNAME and MIXOLOGY_PASSWORD "
            "(creating new ingredients is disabled without them)"
        )

    try:
        response = requests.post(
            f"https://cognito-idp.{region}.amazonaws.com/",
            json={
                "AuthFlow": "USER_PASSWORD_AUTH",
                "ClientId": client_id,
                "AuthParameters": {"USERNAME": username, "PASSWORD": password},
            },
            headers={
                "Content-Type": "application/x-amz-json-1.1",
                "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
            },
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()
    except requests.HTTPError as exc:
        try:
            error = exc.response.json()
            message = error.get("message") or error.get("__type") or str(error)
        except ValueError:
            message = str(exc)
        raise AuthError(
            f"Cognito rejected the login: {message}. Check MIXOLOGY_USERNAME / MIXOLOGY_PASSWORD."
        ) from exc
    except requests.RequestException as exc:
        raise AuthError(f"Could not reach Cognito: {exc}") from exc

    token = result.get("AuthenticationResult", {}).get("AccessToken")
    if not token:
        raise AuthError(f"Cognito returned no access token: {result}")
    return token


def get_token_from_env(
    client_id: str | None = None,
    region: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> str:
    """Resolve credentials from args or environment, then authenticate."""
    return get_token(
        username=username or os.environ.get("MIXOLOGY_USERNAME"),
        password=password or os.environ.get("MIXOLOGY_PASSWORD"),
        client_id=client_id or os.environ.get("MIXOLOGY_CLIENT_ID", DEFAULT_CLIENT_ID),
        region=region or os.environ.get("MIXOLOGY_REGION", DEFAULT_REGION),
    )
