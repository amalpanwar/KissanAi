from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class SupabaseConfig:
    url: str
    anon_key: str
    redirect_url: str = ""


def is_configured(cfg: SupabaseConfig | None) -> bool:
    return bool(cfg and cfg.url.strip() and cfg.anon_key.strip())


def _auth_request(
    cfg: SupabaseConfig,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    bearer: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    url = cfg.url.rstrip("/") + path
    headers = {
        "apikey": cfg.anon_key,
        "Content-Type": "application/json",
    }
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    data = json.dumps(body or {}).encode("utf-8") if body is not None else None
    req = Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urlopen(req, timeout=25) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        return True, json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {"error_description": raw or str(exc)}
        return False, payload
    except URLError as exc:
        return False, {"error_description": str(exc)}
    except Exception as exc:
        return False, {"error_description": str(exc)}


def sign_up(
    cfg: SupabaseConfig,
    *,
    email: str,
    password: str,
    username: str,
    display_name: str,
) -> tuple[bool, dict[str, Any]]:
    body: dict[str, Any] = {
        "email": email,
        "password": password,
        "data": {
            "username": username,
            "display_name": display_name or username,
        },
    }
    if cfg.redirect_url:
        body["email_redirect_to"] = cfg.redirect_url
    return _auth_request(cfg, "POST", "/auth/v1/signup", body)


def sign_in_with_password(cfg: SupabaseConfig, *, email: str, password: str) -> tuple[bool, dict[str, Any]]:
    return _auth_request(
        cfg,
        "POST",
        "/auth/v1/token?grant_type=password",
        {"email": email, "password": password},
    )


def refresh_session(cfg: SupabaseConfig, refresh_token: str) -> tuple[bool, dict[str, Any]]:
    return _auth_request(
        cfg,
        "POST",
        "/auth/v1/token?grant_type=refresh_token",
        {"refresh_token": refresh_token},
    )


def get_user(cfg: SupabaseConfig, access_token: str) -> tuple[bool, dict[str, Any]]:
    return _auth_request(cfg, "GET", "/auth/v1/user", None, bearer=access_token)


def send_password_reset_email(cfg: SupabaseConfig, email: str) -> tuple[bool, dict[str, Any]]:
    body: dict[str, Any] = {"email": email}
    if cfg.redirect_url:
        body["redirect_to"] = cfg.redirect_url
    return _auth_request(cfg, "POST", "/auth/v1/recover", body)


def resend_signup_email(cfg: SupabaseConfig, email: str) -> tuple[bool, dict[str, Any]]:
    body: dict[str, Any] = {"type": "signup", "email": email}
    if cfg.redirect_url:
        body["email_redirect_to"] = cfg.redirect_url
    return _auth_request(cfg, "POST", "/auth/v1/resend", body)


def sign_out(cfg: SupabaseConfig, access_token: str) -> tuple[bool, dict[str, Any]]:
    return _auth_request(cfg, "POST", "/auth/v1/logout", {}, bearer=access_token)

